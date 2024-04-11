import importlib
from config import TRAINERS
from config import base_options
from models.ST_Net import STNET
from utils.data_utils import *
from utils.metrics import *
from clients.base_client import TargetClient
import os
import random
def read_options():
    parser = base_options()
    parsed = parser.parse_args()
    options = parsed.__dict__
    os.environ['PYTHONHASHSEED'] = str(options['seed'])
    np.random.seed(1 + options['seed'])
    torch.manual_seed(12 + options['seed'])
    random.seed(1234 + options['seed'])
    if options['device'].startswith('cuda'):
        torch.cuda.manual_seed_all(123 + options['seed'])
        torch.backends.cudnn.deterministic = True  # cudnn
    trainer_path = 'trainers.%s' % options['algo']
    mod = importlib.import_module(trainer_path)
    trainer_class = getattr(mod, TRAINERS[options['algo']])
    return options, trainer_class

def main():
    options, _ = read_options()
    name=options['target_city']
    train_dataset, test_dataset = getTrainTestDataSet(options, options['target_city'])
    STModel = STNET(options,name)
    c = TargetClient(id=0, name=name, options=options, train_dataset=train_dataset, test_dataset=test_dataset,
                     model=STModel.to(options['device']))
    res = c.target_dataloader()
    print(res)


if __name__ == '__main__':
    main()
