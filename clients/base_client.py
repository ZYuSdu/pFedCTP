from torch.utils.data import DataLoader
from torch import nn
from utils.data_utils import *
from utils.metrics import *
import torch.optim as optim
import torch.nn.functional as F
import copy

class BaseClient(object):

    def __init__(self, id, name, dataset, options, model: nn.Module):
        self.id = id
        self.options = options
        self.name = name
        if self.name == options['target_city']:
            self.notTarget = False
        else:
            self.notTarget = True
        self.inner_lr = options['lr']
        self.num_batch_size = options['batch_size']
        self.device = options['device']
        self.model = model
        self.dataset = dataset
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.inner_lr)
        self.criterion = nn.MSELoss().to(self.device)
        self.mu=0.1





    def train_client_sampleData(self):
        all_loss=0
        for epoch in range(self.options['local_epochs']):
            self.model.train()
            data, _ = self.getSampledData()
            st_output = self.model(data)
            loss = self.calculate_loss(st_output, data.y)
            all_loss+=loss.item()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        return self.name+':'+str(all_loss / self.options['local_epochs'])

    def getSampledData(self):
        batch_size = self.options['batch_size']
        permutation = torch.randperm(self.dataset.x_list.shape[0])
        indices = permutation[0: batch_size]
        x_data = self.dataset.x_list[indices]
        y_data = self.dataset.y_list[indices]
        node_num = self.dataset.A_list.shape[0]
        data = Data(node_num=node_num, x=x_data, y=y_data)
        data.edge_index = self.dataset.edge_index_list
        data.str_init = self.dataset.str_init
        A_wave = self.dataset.A_list.float()
        return data.cuda(), A_wave.cuda()

    def adaptive_set_parameters_list(self, params_list):
        with torch.no_grad():
            avg = []
            for p, d in zip(self.model.parameters(), params_list):
                a = F.cosine_similarity(p.data, d.data, dim=0)
                sim = torch.mean(a)
                avg.append(sim)
                p.data.copy_(p.data + sim * (d.data - p.data))
            # print(sum(avg) / len(avg))

    def calculate_loss(self, out, y):
        loss_predict = self.criterion(out, y)
        return loss_predict

    def set_parameters_list(self, params_list: list):
        """
        :param params_list:
        :return:
        """
        with torch.no_grad():
            for p, d in zip(self.model.parameters(), params_list):
                # 设置参数的值
                p.data.copy_(d.data)

    def set_shared_parameters_list(self, params_list: list):
        """
        :param params_list:
        :return:
        """
        with torch.no_grad():
            ps_pred = [p.data.clone().detach() for p in self.model.stPredictor.parameters()]
            ps_spatial = [p.data.clone().detach() for p in self.model.spatialModel.parameters()]
            ps_gene = [p.data.clone().detach() for p in self.model.stPare.parameters()]

        self.set_parameters_list(params_list)

        with torch.no_grad():
            for p, d in zip(self.model.spatialModel.parameters(), ps_spatial):
                # 设置参数的值
                p.data.copy_(d.data)
            for p, d in zip(self.model.stPredictor.parameters(), ps_pred):
                # 设置参数的值
                p.data.copy_(d.data)
            for p, d in zip(self.model.stPare.parameters(), ps_gene):
                # 设置参数的值
                p.data.copy_(d.data)

    def set_ada_shared_parameters_list(self, params_list: list):
        """
        :param params_list:
        :return:
        """
        with torch.no_grad():
            ps_pred = [p.data.clone().detach() for p in self.model.stPredictor.parameters()]
            ps_spatial = [p.data.clone().detach() for p in self.model.spatialModel.parameters()]
            ps_gene = [p.data.clone().detach() for p in self.model.stPare.parameters()]

        self.adaptive_set_parameters_list(params_list)

        with torch.no_grad():
            for p, d in zip(self.model.spatialModel.parameters(), ps_spatial):
                # 设置参数的值
                p.data.copy_(d.data)
            for p, d in zip(self.model.stPredictor.parameters(), ps_pred):
                # 设置参数的值
                p.data.copy_(d.data)
            for p, d in zip(self.model.stPare.parameters(), ps_gene):
                # 设置参数的值
                p.data.copy_(d.data)

    def get_parameters_list(self) -> list:
        with torch.no_grad():
            ps = [p.data.clone().detach() for p in self.model.parameters()]
        return ps


class TargetClient(BaseClient):
    def __init__(self, id, name, train_dataset, test_dataset, options, model):
        super(TargetClient, self).__init__(id, name, train_dataset, options, model)
        self.train_dataset = train_dataset
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=self.options['batch_size'], shuffle=True,
                                           num_workers=0,
                                           pin_memory=True)
        self.test_dataset = test_dataset
        self.test_dataloader = DataLoader(self.test_dataset, batch_size=self.options['batch_size'], shuffle=True,
                                          num_workers=0,
                                          pin_memory=True)
    def source_dataloader(self):
        all_loss = 0
        count = 0
        self.model.train()
        for epoch in range(self.options['local_epochs']):
            for step, (data) in enumerate(self.train_dataloader):
                data.str_init = self.train_dataset.str_init
                data.edge_index = self.train_dataset.edge_index_list
                data = data.cuda()
                out = self.model(data)
                loss = self.calculate_loss(out, data.y)
                all_loss += loss.item()
                count+=1
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
        return self.name+':'+str(all_loss / count)

    def target_test(self):
        with torch.no_grad():
            self.model.eval()
            for step, (data) in enumerate(self.test_dataloader):
                data.str_init = self.test_dataset.str_init
                data.edge_index = self.test_dataset.edge_index_list
                data = data.cuda()
                out = self.model(data)
                if step == 0:
                    outputs = out
                    y_label = data.y
                else:
                    outputs = torch.cat((outputs, out))
                    y_label = torch.cat((y_label, data.y))
            q_metric = metric_func(outputs, y_label)
        return q_metric

    def target_dataloader(self):
        for epoch in range(self.options['target_epochs']):
            self.model.train()
            for step, (data) in enumerate(self.train_dataloader):
                data.str_init = self.train_dataset.str_init
                data.edge_index = self.train_dataset.edge_index_list
                data = data.cuda()
                out = self.model(data)
                loss = self.calculate_loss(out, data.y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
        q_metric = self.target_test()
        return q_metric


