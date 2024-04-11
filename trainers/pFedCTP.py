from trainers.fedbase import BaseFedarated
from tqdm import *


class pFedCTP(BaseFedarated):

    def __init__(self, options):
        super(pFedCTP, self).__init__(options=options)

    def trainClients(self):
        client_loss=[]
        for c in self.train_clients:
            if self.options['SharePart']:
                c.set_ada_shared_parameters_list(self.latest_model)
            else:
                c.adaptive_set_parameters_list(self.latest_model)
            client_loss.append(c.train_client_sampleData())
        print(client_loss)

    def train(self):
        for round_i in tqdm(range(self.num_rounds)):
            self.select_clients()
            self.trainClients()
            self.aggregate()
        print("---------------------------------")
        for c in self.train_clients:
            if c.name == self.options['target_city']:
                res = c.target_dataloader()
                print(res)