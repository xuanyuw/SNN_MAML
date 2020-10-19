# -----------------------------
# Attention network section
# -----------------------------
# Adapted from https://github.com/BoyuanJiang/matching-networks-pytorch/blob/master/matching_networks.py
import torch
from torch import nn
import numpy as np
from SNN import *

class COS_Similarities(nn.Module):
    def __init__(self):
        super(COS_Similarities, self).__init__()
        self.cos = nn.CosineSimilarity()
    def forward(self, support_set, target_image):
        #TODO: now only support test size = 1
        similarities = []
        for i in range(support_set.size()[0]):
            s_img = torch.gather(support_set, dim=0, index=i)
            dist = self.cos(s_img, target_image)
            similarities.append(dist)
        out = torch.stack(similarities)
        return out

#def COS_Similarities(support_set, target_image):
#    #TODO: now only support test size = 1
#    similarities = []
#    for i in range(support_set.size()[0]):
#        s_img = torch.gather(support_set, dim=0, index=i)
#        cos = nn.CosineSimilarity()
#        dist = cos(s_img, target_image)
#        similarities.append(dist)
#    out = torch.stack(similarities)
#    return out

#class Classification(nn.Module):
#    def __init__(self):
#        super(Classification, self).__init__()
#    def forward(self, similarities, support_labels):
#        softmax = nn.Softmax()
#        a = softmax(similarities)
#        preds = a.bmm(support_labels)
#        return preds

def Classification(similarities, support_labels):
    softmax = nn.Softmax()
    a = softmax(similarities)
    preds = a.bmm(support_labels)
    return preds

class BiDirLSTM(nn.Module):
    def __init__(self, batch_size, input_size, hidden_size=100, num_layers=2):
        super(BiDirLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.input_size = input_size
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, bidirectional=True)
    def forward(self, input):
        hidden_layers = ()
        for i in range(self.num_layers):
            h = torch.randn(self.num_layers * 2, self.batch_size, self.hidden_size)
            hidden_layers = hidden_layers + (h,)
        self.hidden = hidden_layers
        output, self.hidden = self.lstm(input, self.hidden)
        return output

class Matching_Network(nn.Module):
    def __init__(self, batch_size, encoder):
        super(Matching_Network, self).__init__()
        self.batch_size = batch_size
        self.encoder = encoder
        self.bdlstm = BiDirLSTM(batch_size, input_size=encoder.get_out_size())
        self.cos_sim = COS_Similarities()
        #self.classify = Classification()
    def forward(self, support_imgs, support_lbls, target_img):
        # encode both support images and the target image
        s_set = self.encoder(support_imgs)
        s_set = self.bdlstm(s_set.unsqueeze(0))
        t_img = self.encoder(target_img.unsqueeze(0))
        # calculate cos similarities
        sim = self.cos_sim(support_imgs, target_img)
        # prediction
        pred = Classification(sim, support_lbls)
        return pred

