import torch
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
import numpy as np
from sklearn.model_selection import train_test_split
import scipy.sparse as sp
from scipy.linalg import fractional_matrix_power, inv
from torch_geometric.utils import  degree, to_scipy_sparse_matrix
def Structure_init(node_nums,edge_index,n_rw,n_dg):
    A = to_scipy_sparse_matrix(edge_index, num_nodes=node_nums)
    D = (degree(edge_index[0], num_nodes=node_nums) ** -1.0).numpy()
    Dinv = sp.diags(D)
    RW = A * Dinv
    M = RW

    SE = [torch.from_numpy(M.diagonal()).float()]
    M_power = M
    for _ in range(n_rw - 1):
        M_power = M_power * M
        SE.append(torch.from_numpy(M_power.diagonal()).float())
    SE_rw = torch.stack(SE, dim=-1)

    # PE_degree
    g_dg = (degree(edge_index[0], num_nodes=node_nums)).numpy().clip(1, n_dg)
    SE_dg = torch.zeros([node_nums, n_dg])
    for i in range(len(g_dg)):
        SE_dg[i, int(g_dg[i] - 1)] = 1
    str_init = SE_rw
    return str_init

def compute_ppr(adj, alpha=0.2, self_loop=True):
    a = adj
    if self_loop:
        a = a + np.eye(a.shape[0])                                # A^ = A + I_n
    d = np.diag(np.sum(a, 1))                                     # D^ = Sigma A^_ii
    dinv = fractional_matrix_power(d, -0.5)                       # D^(-1/2)
    at = np.matmul(np.matmul(dinv, a), dinv)
    return alpha * inv((np.eye(a.shape[0]) - (1 - alpha) * at))   # a(I_n-(1-a)A~)^-1
def get_diff_func(diff):
    a, b = [], []
    edge_attr = []
    node_feature = None
    matrix = diff
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if (matrix[i][j] > 1):
                a.append(i)
                b.append(j)
    edge = [a, b]
    edge_index = torch.tensor(edge, dtype=torch.long)

    return edge_index, edge_attr, node_feature
def get_attr_func(matrix_path):
    a, b = [], []
    edge_attr = []
    node_feature = None
    matrix = np.load(matrix_path)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if (matrix[i][j] > 0):
                a.append(i)
                b.append(j)
    edge = [a, b]
    edge_index = torch.tensor(edge, dtype=torch.long)

    return edge_index, edge_attr, node_feature
def get_normalized_adj(A):
    A = A + np.diag(np.ones(A.shape[0], dtype=np.float32))
    D = np.array(np.sum(A, axis=1)).reshape((-1,))
    D[D <= 10e-5] = 10e-5    # Prevent infs
    diag = np.reciprocal(np.sqrt(D))
    A_wave = np.multiply(np.multiply(diag.reshape((-1, 1)), A),
                         diag.reshape((1, -1)))
    return A_wave
def generate_dataset(X, num_timesteps_input, num_timesteps_output, means, stds):
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

def get_data(options,city_name):
    path = options['data_path'] + city_name + "/"
    adjPath = path + 'matrix.npy'
    XPath = path + 'dataset.npy'
    str_dim = options['hidden_dim']
    A = np.load(adjPath)
    edge_index, edge_attr, node_feature = get_attr_func(adjPath)
    str_init = Structure_init(A.shape[0], edge_index, n_rw=str_dim, n_dg=str_dim)
    A_list = torch.from_numpy(get_normalized_adj(A))
    X = np.load(XPath)
    print("load:",city_name,",Dataset Shape:",X.shape)
    X = X.transpose((1, 2, 0))
    X = X.astype(np.float32)
    means = np.mean(X, axis=(0, 2))
    X = X - means.reshape(1, -1, 1)
    stds = np.std(X, axis=(0, 2))
    X = X / stds.reshape(1, -1, 1)
    # X = X[:,:,0:500]
    return X,A_list,str_init,edge_index,means,stds

def getTrainTestDataSet(options, city_name,target_days=3):
    X, A_list, str_init, edge_index, means, stds =get_data(options,city_name)
    if city_name == options['target_city']:
        X_Train = X[:, :, :288 * target_days]
        X_train, y_train = generate_dataset(X_Train, options['his_num'], options['pred_num'], means, stds)
        train_dataset = traffic_dataset1(options, city_name, str_init, edge_index, A_list, X_train, y_train)
        X_Test = X[:, :, int(X.shape[2] * 0.8):]
        X_test, y_test = generate_dataset(X_Test, options['his_num'], options['pred_num'], means, stds)
        test_dataset = traffic_dataset1(options, city_name, str_init, edge_index, A_list, X_test, y_test)
        return train_dataset, test_dataset
    else:
        x_inputs, y_outputs = generate_dataset(X, options['his_num'], options['pred_num'], means, stds)
        X_train, X_test, y_train, y_test = train_test_split(x_inputs, y_outputs, test_size=0.2, random_state=options['seed'])
        train_dataset = traffic_dataset1(options, city_name, str_init, edge_index, A_list, X_train, y_train)
        test_dataset = traffic_dataset1(options, city_name, str_init, edge_index, A_list, X_test, y_test)
        return train_dataset, test_dataset


class traffic_dataset(Dataset):
    def __init__(self, options, dataset_name, target_days=3):
        super(traffic_dataset, self).__init__()
        self.options = options
        self.dataset_name = dataset_name
        self.his_num = options['his_num']
        self.pred_num = options['pred_num']
        self.target_days = target_days
        self.target_city_name = options['target_city']
        self.load_data(dataset_name)
    def load_data(self, dataset_name):
        X, A_list, str_init, edge_index, means, stds = get_data(self.options, dataset_name)
        self.str_init = str_init
        self.A_list = A_list
        self.edge_index_list = edge_index
        if dataset_name == self.target_city_name:
            X = X[:, :, :288 * self.target_days]
        x_inputs, y_outputs = generate_dataset(X, self.options['his_num'], self.options['pred_num'], means,stds)
        self.x_list = x_inputs
        self.y_list = y_outputs

    def get(self, index):
        x_data = self.x_list[index: index + 1]
        y_data = self.y_list[index: index + 1]
        node_num = self.A_list.shape[0]
        data_i = Data(node_num=node_num, x=x_data, y=y_data)
        data_i.edge_index = self.edge_index_list
        A_wave = self.A_list
        return data_i, A_wave
    def len(self):
        data_length = self.x_list.shape[0]
        return data_length

class traffic_dataset1(Dataset):
    def __init__(self, options,city_name, str_init, edge_index, A_list, x_inputs, y_outputs):
        super(traffic_dataset1, self).__init__()
        self.options = options
        self.dataset_name = city_name
        self.his_num = options['his_num']
        self.pred_num = options['pred_num']
        self.target_city_name = options['target_city']
        self.str_init = str_init
        self.edge_index_list = edge_index
        self.A_list = A_list
        self.x_list = x_inputs
        self.y_list = y_outputs

    def get(self, index):
        x_data = self.x_list[index: index + 1]
        y_data = self.y_list[index: index + 1]
        node_num = self.A_list.shape[0]
        data_i = Data(node_num=node_num, x=x_data, y=y_data)
        return data_i

    def len(self):
        data_length = self.x_list.shape[0]
        return data_length
