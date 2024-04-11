from trainers.fedbase import BaseFedarated
from models.ST_Net import STNET
from tqdm import *
from utils.data_utils import *
from clients.base_client import TargetClient


class pFedCTP(BaseFedarated):

    def __init__(self, options):
        super(pFedCTP, self).__init__(options=options)

    def setup_clients(self):
        all_clients = []
        client_id = 0
        for d in self.options['data_list']:
            model = STNET(self.options, d).to('cuda')
            train_dataset, test_dataset = getTrainTestDataSet(self.options, d)
            c = TargetClient(id=0, name=d, options=self.options,
                             train_dataset=train_dataset,
                             test_dataset=test_dataset,
                             model=model)
            client_id += 1
            all_clients.append(c)
        return all_clients

    def select_clients(self):
        ind = np.random.permutation(self.num_clients)
        ind = ind[0:len(ind)]
        finial_ind = []
        for i in ind:
            if self.clients[i].name != self.options['target_city']:
                finial_ind.append(i)
            else:
                self.test_client = self.clients[i]
        arryed_cls = np.asarray(self.clients)
        self.train_clients = arryed_cls[finial_ind].tolist()


    def trainClients_Trans(self):
        client_loss=[]
        for c in self.train_clients:
            if self.options['SharePart']:
                c.set_ada_shared_parameters_list(self.latest_model)
            else:
                c.adaptive_set_parameters_list(self.latest_model)
            client_loss.append(c.source_dataloader())
        print(client_loss)


    def fine_tune_target(self):
        d = self.options['target_city']
        model = STNET(self.options, d).to('cuda')
        train_dataset, test_dataset = getTrainTestDataSet(self.options, d)
        c = TargetClient(id=0, name=d, options=self.options,
                                 train_dataset=train_dataset,
                                 test_dataset=test_dataset,
                                 model=model)
        c.set_shared_parameters_list(self.latest_model)
        res = c.target_dataloader()
        return res

    def train(self):
        for round_i in tqdm(range(self.num_rounds)):
            self.select_clients()
            self.trainClients_Trans()
            self.aggregate()
        print("---------------------------------")
        res = self.fine_tune_target()
        print(res)

