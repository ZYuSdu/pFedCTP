import numpy as np
import importlib
import torch
import os
import random
from config import TRAINERS
from config import base_options


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
    options, trainer_class = read_options()
    trainer = trainer_class(options)
    trainer.train()


if __name__ == '__main__':
    main()
