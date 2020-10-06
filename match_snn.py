import argparse
import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np
import double_mnist_loader
from torchvision.transforms import Grayscale, Compose, ToTensor
from Match_net import *
from SNN import *
from double_mnist_loader import *

#parser = argparse.ArgumentParser()
#parser.add_argument('--n_epochs', type=int,default=500)
#parser.add_argument('--batch_size', type=int,default=72)
#parser.add_argument('--n_ways', type=int,default=5)
#parser.add_argument('--support_shots', type=int, default=1)
#parser.add_argument('--test_shots', type=int, default = 5)
#parser.add_argument("--gpu", dest="gpu", action="store_true")
#parser.set_defaults(gpu=False)
#
#args = parser.parse_args()
#
#n_epochs = args.n_epochs
#batch_size = args.batch_size
#n_ways = args.n_ways
#support_shots = args.support_shots
#test_shots = args.test_shots
#train = args.train
#gpu = args.gpu
#test_shots = 1

#TODO: how to back prop?!?!?!
train_set = double_mnist_loader.DoubleMNIST('paired_train.pkl')
val_set = double_mnist_loader.DoubleMNIST('paired_val.pkl')

train_loader = DataLoader(train_set, batch_size=2, shuffle=True, sampler=RandomSampler)
val_loader = DataLoader(val_set, batch_size=2, shuffle=True, sampler=RandomSampler)
loss_fn = nn.MSELoss()


def train():
    mnet = Matching_Network(2, Network())
    mnet.train()
    for i in train_loader:
        #TODO: fix batch thing
        support_imgs = i['support'][1]
        target_img = i['test'][1]
        support_lbls = i['support'][0]
        target_lbl = i['test'][0]
        pred = mnet(support_imgs, support_lbls, target_img)
        loss = loss_fn(pred, target_lbl)
        loss.backward()
    

        



