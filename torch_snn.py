import argparse
import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
from torchmeta.datasets.helpers import doublemnist
from torchmeta.utils.data import BatchMetaDataLoader
from torchvision.transforms import Grayscale, Compose, ToTensor

parser = argparse.ArgumentParser()
parser.add_argument('--n_epochs', type=int,default=500)
parser.add_argument('--batch_size', type=int,default=16)
parser.add_argument('--n_ways', type=int,default=5)
parser.add_argument('--n_shot', type=int,default=1)
parser.add_argument("--train", dest="train", action="store_true")
parser.add_argument("--gpu", dest="gpu", action="store_true")
parser.set_defaults(plot=False, gpu=False, train=True)

args = parser.parse_args()

n_epochs = args.n_epochs
batch_size = args.batch_size
n_ways = args.n_ways
n_shot = args.n_shot
train = args.train
gpu = args.gpu
test_shots = 1

torch.manual_seed(0)
if gpu & torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

# feat for 4-layer structure
img_size = 64
kernel_size = 5
pooling_size = 2
conv_stride = 1
pooling_stride = 2
conv_size = int((64 - kernel_size) / conv_stride) + 1
conv1_out_channel = 20
conv2_out_channel = 50
out_size = int(img_size/(pooling_size * pooling_size))
fc1_in_feat = int(conv2_out_channel * out_size * out_size)
fc1_out_feat = 200
numcat = 100
padding = 2

conv_threshold = 1
pooling_threshold = 0.75
weight_const = 2.0
tau_mem = 100 # time-constant of membrane potential
leak = np.exp(-(1 / tau_mem))

class Spike_generator(torch.autograd.Function):
    @staticmethod
    def forward(self, input):
        self.save_for_backward(input)
        return input.gt(0).float()
    @staticmethod
    def backward(self, grad_output):
        input, = self.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input < 0] = 0
        return grad_input

def LIF_neuron(v_mem, threshold=conv_threshold, leak=leak):
    # approximate LIF using IF first
    exceed_vmem = nn.functional.threshold(v_mem, threshold, 0)
    v_mem = v_mem - exceed_vmem #reset the neurons that already fired
    spike = Spike_generator.apply(exceed_vmem)
    #add in leak effect (leak when forward, no leak in backprop)
    v_mem = leak * v_mem.detach() + v_mem - v_mem.detach() 
    #approx partial aLIF with respect to net = 1/v_threshold
    spike = spike.detach() + torch.true_divide(spike, threshold) - torch.true_divide(spike, threshold).detach()
    return v_mem, spike

def pooling_neuron(v_mem, threshold=pooling_threshold):
    # no leak in pooling layer
    exceed_vmem = nn.functional.threshold(v_mem, threshold, 0)
    v_mem = v_mem - exceed_vmem #reset the neurons that already fired
    spike = Spike_generator.apply(exceed_vmem)
    return v_mem, spike

class Network(nn.Module):
    def __init__(self, n_ways=n_ways, img_size=img_size):
        super(Network, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=conv1_out_channel, kernel_size=kernel_size, 
                              stride=conv_stride, padding=padding, bias=False)
        self.avgpool1 = nn.AvgPool2d(kernel_size=pooling_size, stride=pooling_stride)

        self.conv2 = nn.Conv2d(in_channels=conv1_out_channel, out_channels=conv2_out_channel, 
                              kernel_size=kernel_size, stride=conv_stride, padding=padding, bias=False)
        self.avgpool2 = nn.AvgPool2d(kernel_size=pooling_size, stride=pooling_stride)

        self.fc1 = nn.Linear(fc1_in_feat, fc1_out_feat, bias=False)
        self.fc2 = nn.Linear(fc1_out_feat, numcat, bias=False)

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.in_channels * m.kernel_size[0] * m.kernel_size[1]
                m.weight.data.normal_(0.0, np.sqrt(weight_const/n))
                m.threshold = conv_threshold
            elif isinstance(m, nn.Linear):
                # The weight shape of nn.Linear in PyTorch is (out_features, in_features)!
                fan_in = m.weight.size()[1]
                m.weight.data.normal_(0.0, np.sqrt(weight_const/fan_in))
                m.threshold = conv_threshold

    def forward(self, input, steps=tau_mem, leak=leak):
        vmem_conv1 = Variable(torch.zeros(input.size(0), conv1_out_channel, img_size, img_size), 
                             requires_grad = False)
        vmem_pool1 = Variable(torch.zeros(input.size(0), conv1_out_channel, int(img_size/pooling_size), 
                                          int(img_size/pooling_size)), requires_grad=False)
        vmem_conv2 = Variable(torch.zeros(input.size(0), conv2_out_channel, int(img_size/pooling_size),
                                          int(img_size/pooling_size)), requires_grad=False)
        vmem_pool2 = Variable(torch.zeros(input.size(0), conv2_out_channel, out_size, out_size),
                                          requires_grad=False)
        vmem_fc1 = Variable(torch.zeros(input.size(0), fc1_out_feat), requires_grad=False)
        vmem_fc2 = Variable(torch.zeros(input.size(0), numcat))

        #generate Poisson-distributed spikes
        rand_num = torch.rand(tuple(input.shape))
        poisson_spk = torch.abs(input / 2) > rand_num
        poisson_spk = poisson_spk.float()

        for i in range(steps):
            # conv layer 1
            vmem_conv1 = vmem_conv1 + self.conv1(poisson_spk).int()
            vmem_conv1, spk = LIF_neuron(vmem_conv1)

            # pooling layer 1
            vmem_pool1 = vmem_pool1 + self.avgpool1(spk)
            vmem_pool1, spk = pooling_neuron(vmem_pool1)

            # conv layer 2
            vmem_conv2 = vmem_conv2 + self.conv2(spk)
            vmem_conv2, spk = LIF_neuron(vmem_conv2)

            # pooling layer 2
            vmem_pool2 = vmem_pool2 + self.avgpool2(spk)
            vmem_pool2, spk = pooling_neuron(vmem_pool2)
            
            spk = spk.view(spk.size(0), -1)

            # fully-connected layer 1
            vmem_fc1 = vmem_fc1 + self.fc1(spk)
            vmem_fc1, spk = LIF_neuron(vmem_fc1)

            #fully_connected layer 2
            vmem_fc2 = vmem_fc2 + self.fc2(spk)
            vmem_fc2 = vmem_fc2.detach() * leak + vmem_fc2 - vmem_fc2.detach()
        
        return vmem_fc2 / steps

def train_steps(model, data, label, loss_fc, opt_fc):
    model.train()
    model.zero_grad()
    opt_fc.zero_grad()
    pred = model(data)
    loss = loss_fc(pred, label)
    loss.backward()
    opt_fc.step()
    return loss

def calc_accuracy(model, data, labels):
    model.eval()
    acc = []
    for i in range(len(labels)):
        x = data[i,:,:,:]
        x = x.unsqueeze(0)
        acc.append((model(x).argmax(axis=1) == labels[i]).float())
    return acc


#load dataset
dataset = doublemnist("data", ways=n_ways, shots=n_shot, test_shots=test_shots,
                        meta_train=True, transform=Compose([Grayscale(), ToTensor()]),
                        download=True)
dataloader = BatchMetaDataLoader(dataset, batch_size=batch_size, num_workers=4)

model = Network()
loss_fc = torch.nn.CrossEntropyLoss()
opt_fc = torch.optim.SGD(model.parameters(), lr=1e-3)

#iter_loader=iter(dataloader)
#batch = next(iter_loader)
#train_batch, train_labels = batch['train']
#test_batch, test_labels = batch['test']
#loss = 0
#acc=0
#for i in range(batch_size):
#    tr_d = train_batch[1,:,:,:,:]
#    tr_l = train_labels[i,:]
#    te_d = test_batch[i,:,:,:,:]
#    te_l = test_labels[i,:]
#    loss = train_steps(model, tr_d, tr_l, loss_fc, opt_fc)
#    acc = calc_accuracy(model, te_d, te_l)
#    print('batch test #{}'.format(i))
#    print('loss = {}'.format(loss))
#    print('acc = {}'.format(acc))

# meta learning 
cnt = 0
for batch in dataloader:
    train_batch, train_labels = batch['train']
    test_batch, test_labels = batch['test']
    loss = []
    acc = []
    for i in range(batch_size):
        tr_d = train_batch[1,:,:,:,:]
        tr_l = train_labels[i,:]
        te_d = test_batch[i,:,:,:,:]
        te_l = test_labels[i,:]
        loss.append(train_steps(model, tr_d, tr_l, loss_fc, opt_fc).float())
        acc.append(calc_accuracy(model, te_d, te_l))
    batch_loss = torch.mean(torch.Tensor(loss))
    batch_acc = torch.mean(torch.Tensor(acc))
    cnt += 1
    #if cnt % 100 == 0:
    print('batch #{}'.format(cnt))
    print('loss = {}'.format(batch_loss))
    print('acc = {}'.format(batch_acc))