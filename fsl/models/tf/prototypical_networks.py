"""Prototypical Networks.

Reference:
- Prototypical networks for few-shot learning, Snell et al 2017
"""
import numpy as np
import tensorflow as tf
from tensorflow import keras, nn
from tensorflow.keras import layers

_DATA_FORMAT = 'channels_last'
_NORM_AXIS = -1  # 1 for 'channels_first'

_default_parameters = {
    'conv': {'kernel_size': 3, 'padding':'same', 'data_format':_DATA_FORMAT, 'activation':None},
    'norm': {'axis':_NORM_AXIS}, #, 'momentum':0.5},
    'pool': {'pool_size': 2, 'data_format':_DATA_FORMAT},
    'flat': {'data_format':_DATA_FORMAT},
    'n_conv_block': 4,  # number of convolution blocks
    'hid_channels': 64,  # number of channels in the hidden layers
}

def conv_block(n_channels:int):
    '''A basic convolution block.

    The parameters here follow the original paper.
    '''
    return keras.Sequential([
        layers.Conv2D(n_channels, **_default_parameters['conv']),
        layers.BatchNormalization(**_default_parameters['norm']),
        layers.ReLU(),
        layers.MaxPooling2D(**_default_parameters['pool'])
    ])


def ProtoNet(in_dim:tuple, n_way:int, k_shot:int):
    """Prototypical Network.

    Args
    ----
    in_dim: weight x height x channel
    n_way: number of classes
    k_shot: number of samples per class
    """
    support = layers.Input((None, *in_dim))
    # print(support.shape)
    query = layers.Input((None, *in_dim))
    # query = layers.Input(in_dim)

    _network = keras.Sequential([
        *[conv_block(_default_parameters['hid_channels']) for _ in range(_default_parameters['n_conv_block'])],
        layers.Flatten(**_default_parameters['flat']),
    ])

    xs = _network(tf.reshape(support, (-1, *support.shape[2:])))
    S = tf.reduce_mean(tf.reshape(xs, (n_way, k_shot, -1)), axis=1)

    Q = _network(tf.reshape(query, (-1, *query.shape[2:])))

    dist = tf.reduce_sum((Q[:,None,:]-S[None,:,:])**2, axis=-1)
    output = tf.nn.softmax(-dist, axis=-1)
    # output = -dist  # <- will require `from_logits=True` in the loss function

    return keras.Model(inputs=[query, support], outputs=output, name='ProtoNet')