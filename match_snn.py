import argparse
import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np
import double_mnist_loader
from torchvision.transforms import Grayscale, Compose, ToTensor
from tqdm import tqdm
import SNN
import double_mnist_loader
import Match_net

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
#TODO: fix loader structure batch*ways--> ways so that batch_size=ways
train_set = double_mnist_loader.DoubleMNIST('paired_train.pkl')
val_set = double_mnist_loader.DoubleMNIST('paired_val.pkl')
batch_size = 1

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True)
loss_fn = nn.CrossEntropyLoss()



def train():
    encoder = SNN.Network(batch_size)
    #TODO: change magic num 5 to batch_size
    mnet = Match_net.Matching_Network(5, encoder)
    for i in tqdm(train_loader):
        support_imgs = i['support'][1].squeeze(0)
        target_img = i['test'][1].squeeze(0)
        support_lbls = i['support'][0].squeeze(0)
        target_lbl = i['test'][0].squeeze(0)
        pred = mnet(support_imgs, support_lbls, target_img)
        optimizer = torch.optim.Adam([pred])
        loss = loss_fn(pred, target_lbl)
        acc = sum(torch.eq(target_lbl,pred))/len(pred)
        print('acc=%d, loss=%d' %(acc, loss))
        #optimizer.zero_grad()
        #loss.backward()
        #optimizer.step()

train()

