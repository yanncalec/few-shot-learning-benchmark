"""Prototypical Networks

Reference:
- Prototypical networks for few-shot learning, Snell et al 2017
"""
# from typing import
import functools
import operator

import torch                                        # root package
# torch.manual_seed(time.time())
# from torch.utils.data import Dataset, DataLoader    # dataset representation and loading

import torch.nn as nn                     # neural networks

# import torch.autograd as autograd         # computation graph
# from torch import Tensor                  # tensor node in the computation graph
# import torch.nn.functional as F           # layers, activations and more
# import torch.optim as optim               # optimizers e.g. gradient descent, ADAM, etc.
# # from torch.jit import script, trace       # hybrid frontend decorator and tracing jit

# import torchvision
# from torchvision import datasets, models, transforms     # vision datasets,
#                                                          # architectures &
#                                                          # transforms

# print(torchvision.__version__)

class ProtoNet(nn.Module):
    parameters = {
        'conv': {'kernel_size': 3, 'padding':'same'},
        'pool': {'kernel_size': 2},
        'n_conv_block': 4,  # number of convolution blocks
        'hid_channels': 64,  # number of channels in the hidden layers
        'out_channels': 64,  # number of channels in the output layer
    }

    @classmethod
    def conv_block(cls, in_channels:int, out_channels:int):
        '''A basic convolution block.

        The parameters here are from the original paper.
        '''
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, **cls.parameters['conv']),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(**cls.parameters['pool'])
        )

    def __init__(self, in_dim:tuple):
        super().__init__()

        self.in_dim = in_dim
        hc = self.parameters['hid_channels']
        oc = self.parameters['out_channels']

        # encoder
        self._network = nn.Sequential(
            self.conv_block(in_dim[0], hc),
            *[self.conv_block(hc, hc) for _ in range(self.parameters['n_conv_block']-2)],
            self.conv_block(hc, oc),
            nn.Flatten(),
        )

        # number of features returned by the encoder
        # This will raise error if `in_dim` is too small, due to the maxpool layers
        # self.eval()
        self.n_feature = functools.reduce(operator.mul, list(self._network(torch.rand(1, *in_dim)).shape))

#         _pre_classifier = [
#             nn.Linear(nf, self.n_feature),
#             nn.ReLU(inplace=True),
# #             nn.Linear(2000, 1000),
# #             nn.ReLU(inplace=True),
#         ]

#         self._network = nn.Sequential(
#             *(_pre_network + _pre_classifier)
#         )

    def forward(self, x):
        return self._network(x)


    # def loss(self, sample):
    #     xs = Variable(sample['xs']) # support
    #     xq = Variable(sample['xq']) # query

    #     n_class = xs.size(0)
    #     assert xq.size(0) == n_class
    #     n_support = xs.size(1)
    #     n_query = xq.size(1)

    #     target_inds = torch.arange(0, n_class).view(n_class, 1, 1).expand(n_class, n_query, 1).long()
    #     target_inds = Variable(target_inds, requires_grad=False)

    #     if xq.is_cuda:
    #         target_inds = target_inds.cuda()

    #     x = torch.cat([xs.view(n_class * n_support, *xs.size()[2:]),
    #                    xq.view(n_class * n_query, *xq.size()[2:])], 0)

    #     z = self.encoder.forward(x)
    #     z_dim = z.size(-1)

    #     z_proto = z[:n_class*n_support].view(n_class, n_support, z_dim).mean(1)
    #     zq = z[n_class*n_support:]

    #     dists = euclidean_dist(zq, z_proto)

    #     log_p_y = F.log_softmax(-dists, dim=1).view(n_class, n_query, -1)

    #     loss_val = -log_p_y.gather(2, target_inds).squeeze().view(-1).mean()

    #     _, y_hat = log_p_y.max(2)
    #     acc_val = torch.eq(y_hat, target_inds.squeeze()).float().mean()

    #     return loss_val, {
    #         'loss': loss_val.item(),
    #         'acc': acc_val.item()
    #     }

__all__ = ['ProtoNet']