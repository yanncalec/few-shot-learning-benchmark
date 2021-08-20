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
        # self.n_way = n_way
        # self.k_shot = k_shot

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

        # self.loss = nn.CrossEntropyLoss()

        if self.use_cuda:
            self.cuda()

    def forward(self, x):
        X = Tensor(x)
        return self._network(X.cuda() if self.use_cuda else X)

    def loss_acc(self, xq, xs, yt:list=None):
        """Loss function and accuracy.

        Args
        ----
        xq: query data, of dimension (n_way, q_shot, ...)
        xs: support data, of dimension (n_way, k_shot, ...)

        Returns
        -------
        loss, acc: tensors of the loss function and the accuracy.
        """
        assert xs.shape[0] == xq.shape[0]

        n_way = xs.shape[0]
        q_shot = xq.shape[1]

        # compute prototype of features for support
        S = torch.stack([self.forward(x).mean(axis=0) for x in xs])
        # compute features for query
        Q = self.forward(xq.reshape((-1, *xq.shape[2:])))

        # squared euclidean distance
        dist = ((Q[:,None,:]-S[None,:,:])**2).sum(axis=-1)
        # Gotcha! must use softmax(-d) but not softmax(d)!
        L = F.log_softmax(-dist, -1).reshape((n_way,q_shot,-1))
        yp = L.argmax(axis=-1).flatten()  # predicted labels

        # target labels
        if yt is None:
            yt = torch.arange(n_way)[:,None].expand(-1, q_shot).flatten()
            # yt = np.tile(np.arange(n_way)[:,None], (1,q_shot))
        else:
            yt = torch.tensor(yt)

        # loss_val = -torch.stack([L[n,:,n] for n in range(n_way)]).mean()
        loss_val = F.cross_entropy(-dist, yt)  # yt must be 1d

        if Q.is_cuda:
            acc_val = torch.eq(yp.cuda(), yt.cuda()).float().mean()
        else:
            acc_val = torch.eq(yp, yt).float().mean()

        return loss_val, acc_val


__all__ = ['ProtoNet']