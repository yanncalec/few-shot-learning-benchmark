"""Prototypical Networks, Pytorch implementation.

Reference:
- Prototypical networks for few-shot learning, Snell et al 2017

https://github.com/jakesnell/prototypical-networks
"""

# from typing import
import functools
import operator

import torch                                        # root package
# torch.manual_seed(time.time())
# from torch.utils.data import Dataset, DataLoader    # dataset representation and loading

import torch.nn as nn                     # neural networks
# import torch.autograd as autograd         # computation graph
from torch import Tensor                  # tensor node in the computation graph
import torch.nn.functional as F           # layers, activations and more
# import torch.optim as optim               # optimizers e.g. gradient descent, ADAM, etc.


def euclidean_dist(x, y):
    """Squared euclidean distance between tensors.
    """
    assert x.ndim == y.ndim ==2 and x.shape[1] == y.shape[1]
    # # original implementation:
    # # x: N x D
    # # y: M x D
    # n,d = x.shape
    # m = y.shape[0]
    # x = x.unsqueeze(1).expand(n, m, d)
    # y = y.unsqueeze(0).expand(n, m, d)
    # return torch.pow(x - y, 2).sum(2)  # dimension N x M

    return ((x[:,None,:] - y[None,:,:])**2).sum(-1)  # dimension N x M


class ProtoNet(nn.Module):
    """Prototypical Networks.
    """
    _default_parameters = {
        'conv': {'kernel_size': 3, 'padding':'same'},
        'pool': {'kernel_size': 2},
        'n_conv_block': 4,  # number of convolution blocks
        'hid_channels': 64,  # number of channels in the hidden layers
        'out_channels': 64,  # number of channels in the output layer
    }

    @classmethod
    def conv_block(cls, in_channels:int, out_channels:int):
        '''A basic convolution block.

        The parameters here follow the original paper.
        '''
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, **cls._default_parameters['conv']),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(**cls._default_parameters['pool'])
        )

    def __init__(self, in_dim:tuple, *, use_cuda=False):
        """
        Args
        ----
        in_dim: dimension of the input images (channel, height, width).
        """
        super().__init__()

        self.in_dim = in_dim
        hc = self._default_parameters['hid_channels']
        oc = self._default_parameters['out_channels']

        # encoder
        self._network = nn.Sequential(
            self.conv_block(in_dim[0], hc),
            *[self.conv_block(hc, hc) for _ in range(self._default_parameters['n_conv_block']-2)],
            self.conv_block(hc, oc),
            nn.Flatten(),
        )

        # number of features returned by the encoder
        # This will raise error if `in_dim` is too small, due to the maxpool layers
        # self.eval()
        self.n_feature = functools.reduce(operator.mul, list(self._network(torch.rand(1, *in_dim)).shape))
        self._has_cuda = torch.cuda.is_available()
        self.use_cuda = self._has_cuda and use_cuda

        if self.use_cuda:
            self.cuda()

    def forward(self, x):
        X = Tensor(x)
        return self._network(X.cuda() if self.use_cuda else X)

    @staticmethod
    def loss_acc(Q:Tensor, S:Tensor):
        """Loss function and accuracy.

        Args
        ----
        Q: input feature tensor of query, of dimension (n_query*n_way, n_feature)
        S: input feature tensor of support centroid, of dimension (n_way, n_feature)

        Returns
        -------
        loss, acc: tensors of the loss function and the accuracy.
        """

        n_way = S.shape[0]
        q_shot = Q.shape[0] // n_way

        X = F.log_softmax(((Q[:,None,:]-S[None,:,:])**2).sum(-1), -1)
        L = X.reshape((n_way,q_shot,-1))
        loss_val = -torch.stack([L[n,:,n] for n in range(n_way)]).mean()
        # Yt = np.tile(np.arange(n_way)[:,None], (1,q_shot))  # target labels
        _, Yh = L.max(-1)
        Yt = torch.arange(n_way)[:,None].expand(-1, q_shot)  # target labels

        if Q.is_cuda():
            acc_val = torch.eq(Yh.cuda(), Yt.cuda()).float().mean()
        else:
            acc_val = torch.eq(Yh, Yt).float().mean()

        return loss_val, acc_val


__all__ = ['ProtoNet']