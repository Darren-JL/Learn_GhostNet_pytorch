# used to calculated the params and the flops
import os
import sys
sys.path.append(os.getcwd)

import torch
from ghost_net import ghost_net
from torchsummaryX import summary

model = ghost_net()
summary(model, torch.zeros(3, 3, 224, 224))

