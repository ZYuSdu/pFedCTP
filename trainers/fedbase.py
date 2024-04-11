import abc
from torch import nn, optim

from models.ST_Net import STNET
from utils.data_utils import *
from clients.base_client import BaseClient,TargetClient


class BaseFedarated(abc.ABC):

    def __init__(self, options):
        """
        定义联邦学习的基本的服务器, 这里的模型是在所有的客户端之间共享使用
        :param options: 参数配置
        """
        self.options = options
        self.device = options['device']
        self.clients = self.setup_clients()
        self.num_rounds = options['num_rounds']
        self.num_clients = len(self.clients)
        self.serverModel = STNET(options,'server').to(self.device)
        self.latest_model = [p.data.clone().detach() for p in self.serverModel.parameters()]
        self.optimizer = optim.Adam(self.serverModel.parameters(), lr=self.options['lr'])
        self.criterion = nn.MSELoss().to(self.device)

    def setup_clients(self):
        all_clients = []
        client_id = 0
        print("target city: " + self.options['target_city'])
        for d in self.options['data_list']:
            model = STNET(self.options, d).to('cuda')
            traffic_data = traffic_dataset(self.options, d)
            if d == self.options['target_city']:
                train_dataset, test_dataset = getTrainTestDataSet(self.options, d)
                c = TargetClient(id=0, name=d, options=self.options,
                                     train_dataset=train_dataset,
                                     test_dataset=test_dataset,
                                     model=model)
            else:
                c = BaseClient(id=client_id, name=d, options=self.options, dataset=traffic_data, model=model)
            client_id += 1
            all_clients.append(c)
        return all_clients
    def aggregate(self):
        params = {}
        count=0
        for v in self.latest_model:
            params[count] = torch.zeros_like(v.data)
            count+=1
        index=len(self.train_clients)
        for j in range(len(self.train_clients)):
            # torch.load client model
            clientModel = self.train_clients[j]
            count=0
            for v in (clientModel.get_parameters_list()):
                params[count] += v.data / index
                count+=1
        count=0
        for v in self.latest_model:
            v.data = params[count].data.clone()
            count+=1

    def select_clients(self):
        ind = np.random.permutation(self.num_clients)
        ind = ind[0:len(ind)]
        arryed_cls = np.asarray(self.clients)
        self.train_clients = arryed_cls[ind].tolist()

    @abc.abstractmethod
    def train(self, *args, **kwargs):
        pass
