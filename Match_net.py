# -----------------------------
# Attention network section
# -----------------------------
# Adapted from https://github.com/BoyuanJiang/matching-networks-pytorch/blob/master/matching_networks.py

import torch
from torch import nn
import numpy as np
from SNN import Network

def get_cos_similarities(support_set, target_image):
    #TODO: now only support test size = 1
    similarities = []
    for i in range(support_set.size()[0]):
        s_img = torch.gather(support_set, dim=0, index=i)
        cos = nn.CosineSimilarity()
        dist = cos(s_img, target_image)
        similarities.append(dist)
    out = torch.stack(similarities)
    return out

def calc_pred(similarities, support_labels):
    softmax = nn.Softmax()
    a = softmax(similarities)
    preds = a.bmm(support_labels)
    return preds

class BiDirLSTM(nn.Module):
    def __init__(self, input_size, batch_size, hidden_size=32, num_layers=2):
        super(BiDirLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, bidirectional=True)
    def forward(self, input):
        hidden_layers = ()
        for i in range(self.num_layers):
            h = torch.randn(self.num_layers * 2, self.batch_size, self.hidden_size)
            hidden_layers = hidden_layers + (h,)
        self.hidden = hidden_layers
        output, self.hidden = self.lstm(input, self.hidden)
        return output

class Matching_Network(nn.Module):
    def __init__(self, input_size, batch_size):
        super(Matching_Network, self).__init__()
        self.input_size = input_size
        self.batch_size = batch_size
        self.encoder = Network()
        self.bdlstm = BiDirLSTM(input_size, batch_size)
    def forward(self, support_imgs, support_lbls, target_img):
        # encode both support images and the target image
        s_set = self.encoder(support_imgs)
        s_set = self.bdlstm(s_set)
        t_img = self.encoder(target_img)
        # calculate cos similarities
        sim = get_cos_similarities(support_imgs, target_img)
        # prediction
        pred = calc_pred(sim, support_lbls)
        return pred

