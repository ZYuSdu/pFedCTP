import torch
from torch_geometric.data import Data, Dataset, DataLoader
import numpy as np
from utils import *
import random
def get_normalized_adj(A):
    """
    Returns the degree normalized adjacency matrix.
    """
    A = A + np.diag(np.ones(A.shape[0], dtype=np.float32))
    D = np.array(np.sum(A, axis=1)).reshape((-1,))
    D[D <= 10e-5] = 10e-5    # Prevent infs
    diag = np.reciprocal(np.sqrt(D))
    A_wave = np.multiply(np.multiply(diag.reshape((-1, 1)), A),
                         diag.reshape((1, -1)))
    return A_wave


def generate_dataset(X, num_timesteps_input, num_timesteps_output, means, stds):
    """
    Takes node features for the graph and divides them into multiple samples
    along the time-axis by sliding a window of size (num_timesteps_input+
    num_timesteps_output) across it in steps of 1.
    :param X: Node features of shape (num_vertices, num_features,
    num_timesteps)
    :return:
        - Node features divided into multiple samples. Shape is
          (num_samples, num_vertices, num_features, num_timesteps_input).
        - Node targets for the samples. Shape is
          (num_samples, num_vertices, num_features, num_timesteps_output).
    """
    # Generate the beginning index and the ending index of a sample, which
    # contains (num_points_for_training + num_points_for_predicting) points
    indices = [(i, i + (num_timesteps_input + num_timesteps_output)) for i
               in range(X.shape[2] - (
                num_timesteps_input + num_timesteps_output) + 1)]

    # Save samples
    features, target = [], []
    for i, j in indices:
        features.append(
            X[:, :, i: i + num_timesteps_input].transpose(
                (0, 2, 1)))
        target.append(X[:, 0, i + num_timesteps_input: j]*stds[0]+means[0])

    return torch.from_numpy(np.array(features)), \
           torch.from_numpy(np.array(target))

def get_attr_func(matrix_path):
        a, b = [], []
        edge_attr = []
        node_feature = None
        matrix = np.load(matrix_path)
        # edge_feature_matrix = np.load(edge_feature_matrix_path)
        # node_feature = np.load(node_feature_path)
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                if(matrix[i][j] > 0):
                    a.append(i)
                    b.append(j)
        edge = [a,b]
        edge_index = torch.tensor(edge, dtype=torch.long)

        return edge_index, edge_attr, node_feature

def read_data(options):
    A_list, edge_index_list = {}, {}
    edge_attr_list, node_feature_list = {}, {}
    x_list, y_list = {}, {}
    means_list, stds_list = {}, {}
    print("[INFO]:", options['data_list'])

    for dataset_name in options['data_list']:
        path = options['data_path']+dataset_name+"/"
        adjPath = path+'matrix.npy'
        XPath=path+'dataset.npy'
        A = np.load(adjPath)
        edge_index, edge_attr, node_feature = get_attr_func(adjPath)

        A_list[dataset_name] = torch.from_numpy(get_normalized_adj(A))
        edge_index_list[dataset_name] = edge_index
        edge_attr_list[dataset_name] = edge_attr
        node_feature_list[dataset_name] = node_feature

        X = np.load(XPath)
        X = X.transpose((1, 2, 0))
        X = X.astype(np.float32)
        means = np.mean(X, axis=(0, 2))
        X = X - means.reshape(1, -1, 1)
        stds = np.std(X, axis=(0, 2))
        X = X / stds.reshape(1, -1, 1)
        print(X.shape)
        x_inputs, y_outputs = generate_dataset(X, options['his_num'], options['pred_num'], means,
                                               stds)

        x_list[dataset_name] = x_inputs
        y_list[dataset_name] = y_outputs
        # return [x_list, y_list,A_list,edge_index_list,edge_attr_list,node_feature_list]
        return [x_list, y_list]
# class traffic_dataset(Dataset):
#     def __init__(self, data_args, task_args,  target_days=3):
#         super(traffic_dataset, self).__init__()
#         self.data_args = data_args
#         self.task_args = task_args
#         self.his_num = task_args['his_num']
#         self.pred_num = task_args['pred_num']
#         self.target_days = target_days
#         self.load_data()
#         print("[INFO] Dataset init finished!")
#
#     def load_data(self):
#         self.A_list, self.edge_index_list = {}, {}
#         self.edge_attr_list, self.node_feature_list = {}, {}
#         self.x_list, self.y_list = {}, {}
#         self.means_list, self.stds_list = {}, {}
#         print("[INFO]: dataset: {}".format( self.data_list))
#
#         for dataset_name in self.data_list:
#             A = np.load(self.data_args[dataset_name]['adjacency_matrix_path'])
#             edge_index, edge_attr, node_feature = self.get_attr_func(
#                 self.data_args[dataset_name]['adjacency_matrix_path']
#             )
#
#             self.A_list[dataset_name] = torch.from_numpy(get_normalized_adj(A))
#             self.edge_index_list[dataset_name] = edge_index
#             self.edge_attr_list[dataset_name] = edge_attr
#             self.node_feature_list[dataset_name] = node_feature
#
#             X = np.load(self.data_args[dataset_name]['dataset_path'])
#             X = X.transpose((1, 2, 0))
#             X = X.astype(np.float32)
#             means = np.mean(X, axis=(0, 2))
#             X = X - means.reshape(1, -1, 1)
#             stds = np.std(X, axis=(0, 2))
#             X = X / stds.reshape(1, -1, 1)
#             print(X.shape)
#             x_inputs, y_outputs = generate_dataset(X, self.task_args['his_num'], self.task_args['pred_num'], means,
#                                                    stds)
#
#             self.x_list[dataset_name] = x_inputs
#             self.y_list[dataset_name] = y_outputs
#
#     def get_attr_func(self, matrix_path, edge_feature_matrix_path=None, node_feature_path=None):
#         a, b = [], []
#         edge_attr = []
#         node_feature = None
#         matrix = np.load(matrix_path)
#         # edge_feature_matrix = np.load(edge_feature_matrix_path)
#         # node_feature = np.load(node_feature_path)
#         for i in range(matrix.shape[0]):
#             for j in range(matrix.shape[1]):
#                 if (matrix[i][j] > 0):
#                     a.append(i)
#                     b.append(j)
#         edge = [a, b]
#         edge_index = torch.tensor(edge, dtype=torch.long)
#
#         return edge_index, edge_attr, node_feature
#
#     def get_edge_feature(self, edge_index, x_data):
#         pass
#
#     def get(self, index):
#         """
#         : data.node_num record the node number of each batch
#         : data.x shape is [batch_size, node_num, his_num, message_dim]
#         : data.y shape is [batch_size, node_num, pred_num]
#         : data.edge_index constructed for torch_geometric
#         : data.edge_attr  constructed for torch_geometric
#         : data.node_feature shape is [batch_size, node_num, node_dim]
#         """
#
#         select_dataset = random.choice(self.data_list)
#         batch_size = self.task_args['batch_size']
#         permutation = torch.randperm(self.x_list[select_dataset].shape[0])
#         indices = permutation[0: batch_size]
#         x_data = self.x_list[select_dataset][indices]
#         y_data = self.y_list[select_dataset][indices]
#         node_num = self.A_list[select_dataset].shape[0]
#         data_i = Data(node_num=node_num, x=x_data, y=y_data)
#         data_i.edge_index = self.edge_index_list[select_dataset]
#         data_i.data_name = select_dataset
#         A_wave = self.A_list[select_dataset]
#         return data_i, A_wave
#
#     def get_maml_task_batch(self, task_num):
#         spt_task_data, qry_task_data = [], []
#         spt_task_A_wave, qry_task_A_wave = [], []
#
#         select_dataset = random.choice(self.data_list)
#         batch_size = self.task_args['batch_size']
#
#         for i in range(task_num * 2):
#             permutation = torch.randperm(self.x_list[select_dataset].shape[0])
#             indices = permutation[0: batch_size]
#             x_data = self.x_list[select_dataset][indices]
#             y_data = self.y_list[select_dataset][indices]
#             node_num = self.A_list[select_dataset].shape[0]
#             data_i = Data(node_num=node_num, x=x_data, y=y_data)
#             data_i.edge_index = self.edge_index_list[select_dataset]
#             # data_i.edge_attr = self.edge_attr_list[select_dataset]
#             # data_i.node_feature = self.node_feature_list[select_dataset]
#             data_i.data_name = select_dataset
#             A_wave = self.A_list[select_dataset].float()
#
#             if i % 2 == 0:
#                 spt_task_data.append(data_i.cuda())
#                 spt_task_A_wave.append(A_wave.cuda())
#             else:
#                 qry_task_data.append(data_i.cuda())
#                 qry_task_A_wave.append(A_wave.cuda())
#
#         return spt_task_data, spt_task_A_wave, qry_task_data, qry_task_A_wave
#
#     def len(self):
#         if self.stage == 'source':
#             print("[random permutation] length is decided by training epochs")
#             return 100000000
#         else:
#             data_length = self.x_list[self.data_list[0]].shape[0]
#             return data_length