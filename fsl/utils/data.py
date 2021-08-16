import os
# import functools
# import operator
# from typing import Callable
# from abc import ABC
# import random

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


def imread(fname:str, *, invert=False, dtype=None):
    """Load an image from file to a numpy array.

    Arguments
    ---------
    fname: file name.
    invert: if True the value of image is inverted.
    dtype: data type of the output array.
    """
    with Image.open(fname) as im:
        x = np.array(im, dtype=dtype)
    if x.dtype == bool:
        return ~x if invert else x
    else:
        return np.max(x)-x if invert else x


def load_images_from_folder(rootdir:str, **kwargs):
    """Load images of a folder into a recursive dictionary.
    """
    func = lambda f: imread(f, **kwargs)
    return nested_dict.from_folder(rootdir, func)


def load_images_from_dict(d:dict, rootdir:str='', **kwargs):
    """Load images described of a folder into a recursive dictionary.
    """
    for f in nested_dict.get_paths(d):
        im = imread(os.path.join(rootdir,*f), **kwargs)
        nested_dict.setitem(d, f, im)

    return d


# class Meta_Learning_Sampler(ABC):
#     pass

# class Hierachical_Sampler(Meta_Learning_Sampler):
#     def __init__(self, rootdir:str, ratio_split:float, *, lvl_split:int=0, lvl_task:int=None, lvl_sample:int=None,
#     n_way:int, k_shot:int, q_shot:int):
#         self.rootdir = rootdir
#         self.data_dict = nd.from_folder(self.rootdir)
#         self.lvl_split = lvl_split
#         self.lvl_task = lvl_task
#         self.lvl_sample = lvl_sample
#         # trainval - test split at the level of alphabet
#         self.trainval_dict, self.test_dict = nd.random_split_level(self.data_dict, ratio_split, lvl_split)

#     def sample_task(self):
#         return nd.sample(self.trainval_dict, self.n_way, self.lvl_task)

#     def sample_support_query(self, task:dict, *, split=False):
#         return nd.support_query_sampling(task, self.k_shot, self.q_shot, self.lvl_sample, split=split)
