# GLOBAL PARAMETERS
import argparse
DATASETS = [ 'metr-la', 'pems-bay', 'shenzhen', 'chengdu']
TRAINERS = {'pFedCTP':'pFedCTP','pFedCTP-woF':'pFedCTP','pFedCTP-Trans':'pFedCTP'}

OPTIMIZERS = TRAINERS.keys()
MODEL_CONFIG = {}

def base_options():
    parser = argparse.ArgumentParser()

    parser.add_argument('--algo',
                        help='name of trainer;',
                        type=str,
                        choices=OPTIMIZERS,
                        default='pFedCTP-Trans')
    parser.add_argument('--data_list',
                           help='name of dataset list;',
                           default=DATASETS,
                           type=str)
    parser.add_argument('--target_city',
                        help='target city with scarce data;',
                        default='chengdu',
                        type=str)
    parser.add_argument('--data_path',
                        help='path of dataset;',
                        default="./dataset/",
                        type=str)
    parser.add_argument('--his_num',
                        help='history traffic step;',
                        default=12,
                        type=int)
    parser.add_argument('--pred_num',
                        help='predict traffic step;',
                        default=6,
                        type=int)
    parser.add_argument('--SharePart',
                        default=True,
                        type=bool)
    parser.add_argument('--num_rounds',
                        help='number of rounds to simulate;',
                        type=int,
                        default=2)
    parser.add_argument('--target_epochs',
                        help='number of target client epochs;',
                        type=int,
                        default=2)
    parser.add_argument('--local_epochs',
                        help='number of local client epochs;',
                        type=int,
                        default=4)
    parser.add_argument('--rounds_eval',
                        help='for source clients: each x rounds to eval performances',
                        default=10,
                        type=int)

    parser.add_argument('--gcn_layers',
                        help='the layers of gcn',
                        default=1,
                        type=int)
    parser.add_argument('--message_dim',
                        help='message dim;',
                        default=2,
                        type=int)
    parser.add_argument('--hidden_dim',
                        help='hidden dim;',
                        default=16,
                        type=int)
    parser.add_argument('--wd',
                        help='weight decay parameter;',
                        type=float,
                        default=0.001)
    parser.add_argument('--device',
                        help='device',
                        default='cuda:0',
                        type=str)
    parser.add_argument('--batch_size',
                        help='batch size when clients train on data;',
                        type=int,
                        default=32)
    parser.add_argument('--lr',
                        help='learning rate for inner solver;',
                        type=float,
                        default=0.001)
    parser.add_argument('--outer_lr',
                        help='learning rate for meta-learning',
                        type=float,
                        default=0.001)
    parser.add_argument('--seed',
                        help='seed for randomness;',
                        type=int,
                        default=0)
    return parser

