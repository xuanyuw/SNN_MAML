# -----------------------------
# Spiking network section
# -----------------------------
# Adapted from https://github.com/chan8972/Enabling_Spikebased_Backpropagation/blob/master/src/mnist_LeNet.py

import torch
from torch import nn
from torch.autograd import Variable
import numpy as np


#torch.manual_seed(0)
#if gpu & torch.cuda.is_available():
#    device = torch.device('cuda')
#else:
#    device = torch.device('cpu')

# feat for 4-layer structure
img_size = 64
kernel_size = 5
pooling_size = 2
conv_stride = 1
pooling_stride = 2
#conv_size = int((64 - kernel_size) / conv_stride) + 1
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
        #If a hidden neuron does not fire any spike, the derivative of corresponding neuronal
        # activation is set to zero
        grad_input[input < 0] = 0
        return grad_input

def LIF_neuron(v_mem, threshold=conv_threshold, leak=leak):
    # approximate LIF using IF first
    exceed_vmem = nn.functional.threshold(v_mem, threshold, 0)
    v_mem = v_mem - exceed_vmem #reset the neurons that already fired
    spike = Spike_generator.apply(exceed_vmem)
    #add in leak effect (leak when forward, no leak in backprop)
    v_mem = leak * v_mem.detach() + v_mem - v_mem.detach() 
    #approx partial aIF with respect to net = 1/v_threshold
    spike = spike.detach() + torch.true_divide(spike, threshold) - torch.true_divide(spike, threshold).detach()
    return v_mem, spike

def pooling_neuron(v_mem, threshold=pooling_threshold):
    # no leak in pooling layer
    exceed_vmem = nn.functional.threshold(v_mem, threshold, 0)
    v_mem = v_mem - exceed_vmem #reset the neurons that already fired
    spike = Spike_generator.apply(exceed_vmem)
    return v_mem, spike

def layer_outputs(leak, out, total_out, leak_out, out_history):
    # get output of each layer for gradient approximation
    total_out = total_out + out
    leak_out = leak_out + out * leak
    out_history.append(out)
    return total_out, leak_out, out_history

class Network(nn.Module):
    def __init__(self, batch_size):
        super(Network, self).__init__()
        self.net_out_size = fc1_out_feat
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

    def get_out_size(self):
        return self.net_out_size

    def forward(self, input, steps=tau_mem, leak=leak):
        vmem_conv1 = Variable(torch.zeros(input.size(0), conv1_out_channel, img_size, img_size), 
                             requires_grad=False)
        vmem_pool1 = Variable(torch.zeros(input.size(0), conv1_out_channel, int(img_size/pooling_size), 
                                          int(img_size/pooling_size)), requires_grad=False)
        vmem_conv2 = Variable(torch.zeros(input.size(0), conv2_out_channel, int(img_size/pooling_size),
                                          int(img_size/pooling_size)), requires_grad=False)
        vmem_pool2 = Variable(torch.zeros(input.size(0), conv2_out_channel, out_size, out_size),
                                          requires_grad=False)
        vmem_fc1 = Variable(torch.zeros(input.size(0), fc1_out_feat), requires_grad=False)
        vmem_fc2 = Variable(torch.zeros(input.size(0), numcat))

        #total_out_conv1 = Variable(torch.zeros(input.size(0), conv1_out_channel, img_size, img_size), 
        #                     requires_grad=False)
        #leak_out_conv1 = Variable(torch.zeros(input.size(0), conv1_out_channel, img_size, img_size), 
        #                     requires_grad=False)
        #total_out_conv2 = Variable(torch.zeros(input.size(0), conv2_out_channel, int(img_size/pooling_size),
        #                                  int(img_size/pooling_size)), requires_grad=False)
        #leak_out_conv2 = Variable(torch.zeros(input.size(0), conv2_out_channel, int(img_size/pooling_size),
        #                                  int(img_size/pooling_size)), requires_grad=False)
        #total_out_fc1 = Variable(torch.zeros(input.size(0), fc1_out_feat), requires_grad=False)
        #leak_out_fc1 = Variable(torch.zeros(input.size(0), fc1_out_feat), requires_grad=False)
        #out_history_conv1 = []
        #out_history_conv2 = []
        #out_history_fc1 = []

        #generate Poisson-distributed spikes
        rand_num = torch.rand(tuple(input.shape))
        poisson_spk = torch.abs(input / 2) > rand_num
        poisson_spk = poisson_spk.float()

        # masks for drop-out
        drop_prob = 0.2
        mask_conv1 = Variable(torch.from_numpy(np.random.choice([0, 1], size=(input.size(0), conv1_out_channel, img_size, img_size),
                                        p = [drop_prob, 1-drop_prob])), requires_grad=False)
        mask_conv2 = Variable(torch.from_numpy(np.random.choice([0, 1], size=(input.size(0), conv2_out_channel,
                                                    int(img_size/pooling_size), int(img_size/pooling_size)),
                                        p=[drop_prob, 1 - drop_prob])), requires_grad=False)
        mask_fc1 = Variable(torch.from_numpy(np.random.choice([0, 1], size=(input.size(0), fc1_out_feat),
                                        p = [drop_prob, 1-drop_prob])), requires_grad=False)

        for i in range(steps):
            # conv layer 1
            vmem_conv1 = vmem_conv1 + self.conv1(poisson_spk).int() * (mask_conv1 / (1 - drop_prob)) 
            vmem_conv1, spk = LIF_neuron(vmem_conv1)
    #        total_out_conv1, leak_out_conv1, out_history_conv1 = layer_outputs(leak, spk, total_out_conv1, leak_out_conv1, out_history_conv1)

            # pooling layer 1
            vmem_pool1 = vmem_pool1 + self.avgpool1(spk)
            vmem_pool1, spk = pooling_neuron(vmem_pool1)

            # conv layer 2
            vmem_conv2 = vmem_conv2 + self.conv2(spk) * (mask_conv2 / (1 - drop_prob)) 
            vmem_conv2, spk = LIF_neuron(vmem_conv2)
    #        total_out_conv2, leak_out_conv2, out_history_conv2 = layer_outputs(leak, spk, total_out_conv2, leak_out_conv2, out_history_conv2)


            # pooling layer 2
            vmem_pool2 = vmem_pool2 + self.avgpool2(spk)
            vmem_pool2, spk = pooling_neuron(vmem_pool2)
            
            spk = spk.view(spk.size(0), -1)
            # fully-connected layer 1
            vmem_fc1 = vmem_fc1 + self.fc1(spk) * (mask_fc1 / (1 - drop_prob)) 
            vmem_fc1, spk = LIF_neuron(vmem_fc1)
            #total_out_fc1, leak_out_fc1, out_history_fc1 = layer_outputs(leak, spk, total_out_fc1, leak_out_fc1, out_history_fc1)

            return vmem_fc1


        #    #fully_connected layer 2
        #    vmem_fc2 = vmem_fc2 + self.fc2(spk)
        #    vmem_fc2 = vmem_fc2.detach() * leak + vmem_fc2 - vmem_fc2.detach()
        #
        #return vmem_fc2 / steps
        #,total_out_conv1, leak_out_conv1, out_history_conv1,
        #total_out_conv2, leak_out_conv2, out_history_conv2,
        #total_out_fc1, leak_out_fc1, out_history_fc1

# TODO: find a way to add manual bp
