import torch
from torch import nn
from copy import deepcopy
from tqdm import tqdm
import scipy.sparse as sp
from torch import optim
from torch.nn import MultiheadAttention
from torch_geometric.nn import GATConv, GCNConv
import torch
from torch import nn

from torch.nn import functional as F


class Generate_Linear(nn.Module):
    def __init__(self,hidden_dim):
        super(Generate_Linear, self).__init__()
        self.w1_linear = nn.Linear(hidden_dim, hidden_dim)
        self.b_linear = nn.Linear(hidden_dim, 1)
    def forward(self, meta_knowledge):
        w_meta = self.w1_linear(meta_knowledge)
        b_meta = self.b_linear(meta_knowledge)
        output = w_meta*meta_knowledge+b_meta
        return output
class TemporalModel(nn.Module):
    def __init__(self, options):
        super(TemporalModel, self).__init__()
        self.options = options
        self.hidden_dim = options['hidden_dim']
        self.message_dim = options['message_dim']
        self.his_num = options['his_num']
        self.tp_learner = nn.GRU(self.message_dim, 1, batch_first=True)
        self.tp_attention = MultiheadAttention(embed_dim=self.hidden_dim, num_heads=4)
        self.gru_out = torch.nn.Linear(self.his_num, self.hidden_dim)


    def forward(self, data):
        batch_size, node_num, his_len, message_dim = data.x.shape
        # tp
        self.tp_learner.flatten_parameters()
        x_bn_h_2 = torch.reshape(data.x, (batch_size * node_num, his_len, message_dim))
        tp_gru_h, _ = self.tp_learner(x_bn_h_2)
        tp_gru_h = tp_gru_h.squeeze(-1)
        tp_gru_h = torch.reshape(tp_gru_h, (batch_size, node_num, his_len))
        tp_gru_h = self.gru_out(tp_gru_h)
        attn_output, attn_output_weights = self.tp_attention(tp_gru_h, tp_gru_h, tp_gru_h)
        tem_h=torch.cat((tp_gru_h, attn_output), -1)
        return tp_gru_h,tem_h

class SpatialModel(nn.Module):
    def __init__(self, options):
        super(SpatialModel, self).__init__()
        self.options = options
        self.message_dim = options['message_dim']
        self.his_num = options['his_num']
        self.nlayer = options['gcn_layers']
        self.hidden_dim = options['hidden_dim']
        self.str_init_dim = self.hidden_dim
        self.embedding_s = torch.nn.Linear(self.str_init_dim, 2 * self.hidden_dim)
        self.graph_convs_s_gcn = torch.nn.ModuleList()
        for l in range(self.nlayer):
            self.graph_convs_s_gcn.append(GCNConv(2 * self.hidden_dim, 2 * self.hidden_dim))
        self.embedding_s_out = torch.nn.Linear(2 * self.hidden_dim, self.options['pred_num'])
        self.sp_learner = GATConv(self.hidden_dim, self.options['pred_num'], 3, False, dropout=0.1)


    def forward(self, data,tep_out):
        batch_size, node_num, his_len, message_dim = data.x.shape

        s = self.embedding_s(data.str_init)
        for i in range(self.nlayer):
            s = self.graph_convs_s_gcn[i](s, data.edge_index)
            s = torch.tanh(s)

        str_emb = torch.repeat_interleave(s.unsqueeze(0), batch_size, dim=0)
        str_emb = self.embedding_s_out(str_emb)
        tep_bn_hm = torch.reshape(tep_out, (batch_size * node_num, self.hidden_dim))
        spa_h = self.sp_learner(tep_bn_hm, data.edge_index)
        spa_h = torch.reshape(spa_h, (batch_size, node_num, self.options['pred_num']))
        return str_emb,spa_h

class SharedModel(nn.Module):
    def __init__(self, options):
        super(SharedModel, self).__init__()
        self.options = options
        self.hidden_dim = options['hidden_dim']
        self.pred_num=options['pred_num']
        self.message_dim = options['message_dim']
        self.temporalModel = TemporalModel(options)
        self.tem_predictor1 = nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        self.tem_predictor2 = nn.Linear(self.hidden_dim,self.pred_num)
    def forward(self, data):
        tp_gru_h,tem_h = self.temporalModel(data)
        output = self.tem_predictor1(tem_h)
        output = self.tem_predictor2(output)
        return tp_gru_h,output

class STNET(nn.Module):
    def __init__(self, options,name):
        super(STNET, self).__init__()
        self.options = options
        self.message_dim = options['message_dim']
        self.name = name
        self.pred_num = options['pred_num']
        self.spatialModel = SpatialModel(options)
        self.shareModel = SharedModel(options)
        self.stPredictor = nn.Linear(self.pred_num*3,self.pred_num)
        self.stPare = Generate_Linear(self.pred_num)
    def forward(self, data):
        tp_gru_h,tep_output=self.shareModel(data)
        str_emb,spa_output = self.spatialModel(data,tp_gru_h)
        sp_pa=self.stPare(spa_output)
        st_output=self.stPredictor(torch.cat((str_emb,sp_pa,tep_output),-1))
        return st_output


