import os
import subprocess

os.environ['PYTHONIOENCODING'] = 'utf-8'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

seed = 77

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True

from utils.parser import get_parser
from utils.logger import get_logger

parser = get_parser()
option = parser.parse_args()

root_folder = os.path.join('result', option.name)

subprocess.run('mkdir -p %s' % root_folder, shell = True)

logger = get_logger(option.name, os.path.join(root_folder, 'main.log'))

from loaders.loader import get_loader
from modules.module import get_module

from optimizers.optimizer1 import get_optimizer as get_optimizer1
from optimizers.optimizer2 import get_optimizer as get_optimizer2
from optimizers.optimizer3 import get_optimizer as get_optimizer3
from optimizers.optimizer4 import get_optimizer as get_optimizer4
from optimizers.optimizer5 import get_optimizer as get_optimizer5

from utils.misc import train, valid, mse_loss, cross_entropy, plot_loss

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

logger.info('prepare loader')

train_loader, valid_loader = get_loader(option)

logger.info('prepare module')

module = get_module(option)
module = module.to (device)

logger.info('prepare envs')

criterion = torch.nn.functional.cross_entropy

if option.optim_type == 1:
    optimizer = get_optimizer1(module.parameters(), option)
if option.optim_type == 2:
    optimizer = get_optimizer2(module.parameters(), option)
if option.optim_type == 3:
    optimizer = get_optimizer3(module.parameters(), option)
if option.optim_type == 4:
    optimizer = get_optimizer4(module.parameters(), option)
if option.optim_type == 5:
    optimizer = get_optimizer5(module.parameters(), option)

logger.info('start train')

train_loss = []
valid_loss = []
for epoch in range(option.num_epochs):
    train_info = train(train_loader, module, criterion, optimizer, device)
    valid_info = valid(valid_loader, module, criterion, optimizer, device)
    logger.info(
        'epoch: %d, train_loss: %.7f, valid_loss: %.7f, valid_macro_f1: %.7f, valid_micro_f1: %.7f' %
        (epoch, train_info['loss'], valid_info['loss'], valid_info['macro_f1'], valid_info['micro_f1'])
    )
    train_loss.append(train_info['loss'])
    valid_loss.append(valid_info['loss'])

plot_loss(train_loss, valid_loss, os.path.join(root_folder, 'optim_' + str(option.optim_type) + '.jpg'))
