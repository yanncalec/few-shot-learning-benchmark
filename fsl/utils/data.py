import os
# import functools
# import operator
from typing import Callable
# from abc import ABC
import random

import numpy as np

# from skimage.io import imread
# from skimage.transform import resize
# from imageio import imread
from PIL import Image

import logging
Logger = logging.getLogger('fsl.utils.data')

from . import nested_dict as nd


def regularize_filename(d:str):
    """Regularize a filename format by removing unnecessary characters.
    """
    d = d.strip(' ').rstrip(' ').rstrip(os.sep)
    s = d.split(os.sep)
    r = os.path.join(*s)
    return os.sep+r if s[0] == '' else r


def imread(fname:str, *, invert=False, dtype=None, dim:tuple=None, dfmt:str='channels_last'):
    """Load an image from file to a numpy array.

    Arguments
    ---------
    fname: file name.
    invert: if True the value of image is inverted.
    dtype: data type of the output array.
    """
    with Image.open(fname) as im:
        if dim:
            im = im.resize(dim)
        x = np.atleast_3d(np.array(im, dtype=dtype))  # H x W x C
        if dfmt == 'channels_first':  # C x W x H
            x = x.transpose((2,1,0))

    if x.dtype == bool:
        return ~x if invert else x
    else:
        return np.max(x)-x if invert else x


def load_images_from_folder(rootdir:str, **kwargs):
    """Load images of a folder into a recursive dictionary.
    """
    func = lambda f: imread(f, **kwargs)
    return nd.from_folder(rootdir, func)


def load_images_from_dict(d:dict, rootdir:str, paths:list=None, **kwargs):
    """Load images described of a folder into a recursive dictionary, inplace operation.
    """
    if not paths:
        _paths = nd.get_paths(d)
    else:
        _paths = nd.get_paths(nd.sub_dict(d, paths))  # a trick s.t. the paths are complete, e.g. down to the leaves

    images = []
    for f in _paths:
        im = imread(os.path.join(rootdir, *f), **kwargs)
        nd.setitem(d, f, im)
        images.append(im)

    return images, _paths
