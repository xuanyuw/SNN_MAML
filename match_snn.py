import argparse
import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np
import double_mnist_loader
from torchvision.transforms import Grayscale, Compose, ToTensor

parser = argparse.ArgumentParser()
parser.add_argument('--n_epochs', type=int,default=500)
parser.add_argument('--batch_size', type=int,default=72)
parser.add_argument('--n_ways', type=int,default=5)
parser.add_argument('--support_shots', type=int, default=1)
parser.add_argument('--test_shots', type=int, default = 5)
parser.add_argument("--gpu", dest="gpu", action="store_true")
parser.set_defaults(gpu=False)

args = parser.parse_args()

n_epochs = args.n_epochs
batch_size = args.batch_size
n_ways = args.n_ways
support_shots = args.support_shots
test_shots = args.test_shots
train = args.train
gpu = args.gpu
test_shots = 1



