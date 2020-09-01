# Code modified from https://github.com/BindsNET/bindsnet/blob/master/examples/mnist/conv_mnist.py
#@article{lee2020enabling,
#  doi={10.3389/fnins.2020.00119}
#  title={Enabling Spike-based Backpropagation for Training Deep Neural Network Architectures},
#  author={Lee, Chankyu and Sarwar, Syed Shakib and Panda, Priyadarshini and Srinivasan, Gopalakrishnan and Roy, Kaushik},
#  journal={Frontiers in Neuroscience},
#  volume={14},
#  pages={119},
#  year={2020},
#  url={https://www.frontiersin.org/article/10.3389/fnins.2020.00119},
#  publisher={Frontiers in Neuroscience}
#}

import argparse
from typing import Union, Tuple
import torch
from torch.nn.modules.utils import _pair
import torch.nn.functional as F
import numpy as np
from torchmeta.datasets.helpers import doublemnist
from torchmeta.utils.data import BatchMetaDataLoader

from time import time as t
from tqdm import tqdm

from bindsnet.encoding import PoissonEncoder
from bindsnet.network import Network
from bindsnet.learning import PostPre
from bindsnet.network.monitors import Monitor
from bindsnet.network.nodes import DiehlAndCookNodes, Input, Nodes
from bindsnet.network.topology import Conv2dConnection, Connection, AbstractConnection
from bindsnet.analysis.plotting import (
    plot_input,
    plot_spikes,
    plot_conv2d_weights,
    plot_voltages,
)

parser = argparse.ArgumentParser()
parser.add_argument('--n_epochs', type=int,default=500)
parser.add_argument('--batch_size', type=int,default=72)
parser.add_argument('--n_ways', type=int,default=5)
parser.add_argument('--n_shot', type=int,default=1)
parser.add_argument("--train", dest="train", action="store_true")
parser.add_argument("--plot", dest="plot", action="store_true")
parser.add_argument("--gpu", dest="gpu", action="store_true")
parser.set_defaults(plot=False, gpu=False, train=True)

args = parser.parse_args()

n_epochs = args.n_epochs
batch_size = args.batch_size
n_ways = args.n_ways
n_shot = args.n_shot
train = args.train
plot = args.plot
gpu = args.gpu

torch.manual_seed(0)
if gpu & torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

# Implement the AvgPooling Connection
class AvgPool2dConnection(AbstractConnection):
    def __init__(
        self,
        source: Nodes,
        target: Nodes,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int]] = 0,
        dilation: Union[int, Tuple[int, int]] = 1,
        **kwargs
    ) -> None:
        # language=rst
        """
        Instantiates a ``MaxPool2dConnection`` object.
        :param source: A layer of nodes from which the connection originates.
        :param target: A layer of nodes to which the connection connects.
        :param kernel_size: Horizontal and vertical size of convolutional kernels.
        :param stride: Horizontal and vertical stride for convolution.
        :param padding: Horizontal and vertical padding for convolution.
        :param dilation: Horizontal and vertical dilation for convolution.
        Keyword arguments:
        :param decay: Decay rate of online estimates of average firing activity.
        """
        super().__init__(source, target, None, None, 0.0, **kwargs)

        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)

        self.register_buffer("firing_rates", torch.zeros(source.shape))

    def compute(self, s: torch.Tensor) -> torch.Tensor:
        # language=rst
        """
        Compute avg-pool pre-activations given spikes using online firing rate
        estimates.
        :param s: Incoming spikes.
        :return: Incoming spikes multiplied by synaptic weights (with or without
            decaying spike activation).
        """
        self.firing_rates -= self.decay * self.firing_rates
        self.firing_rates += s.float().squeeze()

        _, indices = F.avg_pool2d(
            self.firing_rates,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            return_indices=True,
        )

        return s.take(indices).float()

    def update(self, **kwargs) -> None:
        # language=rst
        """
        Compute connection's update rule.
        """
        super().update(**kwargs)

    def normalize(self) -> None:
        # language=rst
        """
        No weights -> no normalization.
        """
        pass

    def reset_state_variables(self) -> None:
        # language=rst
        """
        Contains resetting logic for the connection.
        """
        super().reset_state_variables()

        self.firing_rates = torch.zeros(self.source.shape)

# feat for conv, 4-layer structure
kernel_size = 5
pooling_size = 2
conv_stride = 1
pooling_stride = 2
conv_size = int((64 - kernel_size) / conv_stride) + 1

# Building network (4-layer)
network = Network()
input_layer = Input(n=4096, shape=(1, 64, 64), traces=True)
n_filters_1 = 1
n_filters_2 = 20
theta_plus: float = 0.05
tc_theta_decay: float = 1e7

conv_layer_1 = DiehlAndCookNodes(
    n=n_filters_1 * conv_size * conv_size,
    shape=(n_filters_1, conv_size, conv_size),
    rest=-65.0,
    reset=-60.0,
    thresh=-52.0,
    refrac=5,
    tc_decay=100.0,
    tc_trace=20.0,
    theta_plus=theta_plus,
    tc_theta_decay=tc_theta_decay,
    trace=True
)

conv_layer_2 = DiehlAndCookNodes(
    n=n_filters_2 * conv_size * conv_size,
    shape=(n_filters_2, conv_size, conv_size),
    rest=-65.0,
    reset=-60.0,
    thresh=-52.0,
    refrac=5,
    tc_decay=100.0,
    tc_trace=20.0,
    theta_plus=theta_plus,
    tc_theta_decay=tc_theta_decay,
    trace=True
)

avg_pooling_1 = AvgPool2dConnection(
    source=input_layer,
    target=conv_layer_1,
    kernel_size=pooling_size,
    stride=pooling_stride
)

avg_pooling_2 = AvgPool2dConnection(
    source=conv_layer_1,
    target=conv_layer_2,
    kernel_size=pooling_size,
    stride=pooling_stride
)

network.add_layer(input_layer, name='X')
network.add_layer(conv_layer_1, name='Y')
network.add_layer(conv_layer_2, name='Z')
network.add_connection(avg_pooling_1, source='X', target='Y')
network.add_connection(avg_pooling_2, source='Y', target='Z')

time = 250

voltage_monitor = Monitor(network.layers['Z'], ['v'], time=time)
network.add_monitor(voltage_monitor, name="output_voltage")

# Directs network to GPU
if gpu:
    network.to("cuda")


# load meta-training data
task_batch_size = 5
niterations = 100000
n_ways = 5
inner_epochs = 5
meta_eval_epochs = 1
eval_epochs = 10
inner_batch_size = 10
outerstepsize0 = .1

train_dataset = doublemnist('data', shots=10, ways=n_ways, shuffle=True, meta_split = "train", test_shots=15, download=False)
test_dataset = doublemnist('data', shots=1, ways=n_ways, shuffle=True, meta_split = "val", test_shots=15, download=False)
train_dataloader = BatchMetaDataLoader(train_dataset, batch_size=task_batch_size, num_workers=4)
test_dataloader = BatchMetaDataLoader(test_dataset, batch_size=1, num_workers=4)
train_dataloader_iter = iter(train_dataloader)
test_dataloader_iter = iter(test_dataloader)

#meta-test evaluation
mbatch = next(test_dataloader_iter)
mdata, mtargets = mbatch["train"]
mdata_, mtargets_ = mbatch["test"]

