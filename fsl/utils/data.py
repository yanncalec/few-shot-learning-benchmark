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


def imread(fname:str, *, invert=False, dtype=None, dim:tuple=None):
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
        x = np.atleast_3d(np.array(im, dtype=dtype)).transpose((2,0,1))  # make the first dimension the channel

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
        paths = nd.get_paths(d)

    for f in paths:
        im = imread(os.path.join(rootdir, *f), **kwargs)
        nd.setitem(d, f, im)
    return d


def leaves_to_array(d:dict, lvl:int=None):
    def _leaves_to_array(d:dict):
        res = [nd.getitem(d, p) for p in nd.get_paths(d)]
        return np.asarray(res)

    if lvl is None:
        return _leaves_to_array(d)
    else:
        paths = nd.get_paths(d, lvl)
        res = []
        for p in paths:
            foo = nd.getitem(d, p)
            if type(foo) is dict:
                res.append(_leaves_to_array(foo))
            else:
                res.append(foo)
        return np.asarray(res)
        # return np.concatenate(res)


def element_to_index(a:list):
    try:
        b = list(set(a))  # only works for hashable elements
    except:
        b = []
        for u in a:
            if u not in b:
                b.append(u)
    d = []
    for l in a:
        for n,p in enumerate(b):
            if p==l:
                break
        d.append(n)
    return d


class Meta_Sampler:
    def __init__(self, data_dict, rootdir:str, *, \
                 lvl_task:int, lvl_sample:int=None, \
                 n_way:int, k_shot:int, q_shot:int=None, cached=True):
        """Sampler for meta-learning.

        Args
        ----
        data_dict: nested dictionary representing the data structure
        rootdir: root directory
        lvl_task: level of task sampling
        lvl_sample: level of data sampling, optional
        n_way: number of classes
        k_shot: number of samples per class for training data
        q_shot: number of samples per class for query data
        """
        self.data_dict = data_dict  # nested dictionary representing the data structure
        self.rootdir = rootdir
        self.lvl_task = lvl_task  # level of task sampling
        self.lvl_sample = lvl_sample  # level of data sampling
        self.n_way = n_way
        self.k_shot = k_shot
        self.q_shot = q_shot
        self.cached = cached
        self.n_leaves = nd.count_leaves(self.data_dict)
        self.n_leafnodes = nd.count_leafnodes(self.data_dict)
#         self.n_task_max = comb(self.n_leafnodes, self.n_way)
        self.paths = nd.get_paths(self.data_dict)
        self.sep = '|'

    def sample_task(self):
        """Sample a random task.
        """
        return nd.sample(self.data_dict, self.n_way, self.lvl_task)

    def sample_support_query(self, task:dict, *, split=False, offclass=False, func:Callable=imread, **kwargs):
        """Sample a support dataset and a query dataset.
        """
        def _load_dict(d:dict):
            paths = nd.get_paths(d)
            # paths = nd.get_paths(nd.sub_dict(self.data_dict, nd.get_paths(d)))  # a trick s.t. the paths are complete, e.g. down to the leaves
            for p in paths:
                val = nd.getitem(self.data_dict, p)
                if val is None or not self.cached:
                    val = func(os.path.join(self.rootdir, *p), **kwargs)
                    # if transform:
                    #     val = transform(val)
                    nd.setitem(self.data_dict, p, val)
            return nd.sub_dict(self.data_dict, paths)

        sd, qd = nd.support_query_sampling(task, self.k_shot, self.q_shot, self.lvl_sample, split=split)

        sa = leaves_to_array(_load_dict(sd), -1)  # 5-dims array: n_way x k_shot x n_channels x height x width
        # sa = leaves_to_array(_load_dict(sd))  # 4-dims array: (n_way x k_shot) x n_channels x height x width
        sl = [self.sep.join(p) for p in nd.get_paths(sd, -1)]
        qa = leaves_to_array(_load_dict(qd), -1)  # n_way x q_shot x n_channels x height x width
        ql = [self.sep.join(p) for p in nd.get_paths(qd, -1)]

        if offclass:
            # off-class samples
            tpaths = ['|'.join(p) for p in nd.get_paths(task,-1)]
            opaths = []
            for _ in range(qa[-1].shape[0]):
                while True:  # with replacement
                    p = random.choice(self.paths)
                    if self.sep.join(p) not in tpaths:
                        opaths.append(p)
                        break
            od = nd.sub_dict(self.data_dict, opaths)

            # much slower
            # ctask = nd.sub_dict(self.data_dict, nd.get_paths(task), compl=True)
            # od = nd.sample(ctask, qa[-1].shape[0], replace=True)

            qa = np.concatenate([qa, leaves_to_array(_load_dict(od))[None,:]])
            ql.append('')

        return (sa, sl), (qa, ql)

