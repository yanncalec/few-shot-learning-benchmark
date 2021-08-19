import os
import random
import math

# from abc import ABC
from typing import Callable
from fsl.utils import data
from fsl.utils import nested_dict as nd


class MetaSampler:
    """Sampler for meta-learning.
    """

    def __init__(self, data_dict, rootdir:str, *, \
                 lvl_task:int, lvl_sample:int=None, \
                 n_way:int, k_shot:int, q_shot:int=None, split=False, local=False, \
                 cached=True, func:Callable=data.imread):
        """
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
        self.n_leaves = nd.count_leaves(self.data_dict)
        self.n_leafnodes = nd.count_leafnodes(self.data_dict)
#         self.n_task_max = comb(self.n_leafnodes, self.n_way)
        self.split = split
        self.local = local
        self.cached = cached
        if local:
            self.paths = nd.get_paths(self.data_dict, self.lvl_task-1)
        else:
            self.paths = nd.get_paths(self.data_dict, self.lvl_task)
        # self.paths = nd.get_paths(self.data_dict, self.lvl_task)
        # self.paths = nd.get_paths(self.data_dict)
        self.func = func
        self.sep = '|'

    def __len__(self):
        if self.local:
            return len(self.paths)
        else:
            return math.floor(len(self.paths)/self.n_way)

    # def __getitem__(self, idx):
    #     if self.local:
    #         X, paths = data.load_images_from_dict(self.data_dict, self.rootdir, [self.paths[idx]], **self.kwargs_imread)
    #         # idx = list(range(len(X)))
    #         # random.shuffle(idx)
    #         S = np.asarray(random.choices(X, k=self.k_shot)) if self.k_shot else np.asarray(X)
    #         Q = np.asarray(random.choices(X, k=self.q_shot))
    #         return [Q, S]
    #     else:
    #         # _paths = [self.paths[n] for n in range(idx*self.n_way, (idx+1)*self.n_way) if n<len(self.paths)]
    #         # X = data.load_images_from_dict(self.data_dict, self.rootdir, _paths, **self.kwargs_imread)
    #         raise NotImplementedError()

    def shuffle(self):
        random.shuffle(self.paths)

    def load_dict(self, d:dict):
        # paths = nd.get_paths(d)
        paths = nd.get_paths(nd.sub_dict(self.data_dict, nd.get_paths(d)))  # a trick s.t. the paths are complete, e.g. down to the leaves
        for p in paths:
            val = nd.getitem(self.data_dict, p)
            if val is None or not self.cached:
                val = self.func(os.path.join(self.rootdir, *p))
                # if transform:
                #     val = transform(val)
                nd.setitem(self.data_dict, p, val)
        return nd.sub_dict(self.data_dict, paths)

    def get_task(self, idx):
        if self.local:
            _paths = [self.paths[idx]]
        else:
            _paths = [self.paths[n] for n in range(idx*self.n_way, (idx+1)*self.n_way) if n<len(self.paths)]

        return nd.sub_dict(self.data_dict, _paths)

    def sample_support_query(self, task:dict, *, split=False, offclass=False):
        """Sample a support dataset and a query dataset.
        """
        sd, qd = nd.support_query_sampling(task, self.k_shot, self.q_shot, self.lvl_sample, split=split)

        sa = nd.leaves_to_array(self.load_dict(sd), -1)  # 5-dims array: n_way x k_shot x n_channels x height x width
        # sa = nd.leaves_to_array(self.load_dict(sd))  # 4-dims array: (n_way x k_shot) x n_channels x height x width
        sl = [self.sep.join(p) for p in nd.get_paths(sd, -1)]
        qa = nd.leaves_to_array(self.load_dict(qd), -1)  # n_way x q_shot x n_channels x height x width
        ql = [self.sep.join(p) for p in nd.get_paths(qd, -1)]

        # if offclass:
        #     # off-class samples
        #     tpaths = ['|'.join(p) for p in nd.get_paths(task,-1)]
        #     opaths = []
        #     for _ in range(qa[-1].shape[0]):
        #         while True:  # with replacement
        #             p = random.choice(self.paths)
        #             if self.sep.join(p) not in tpaths:
        #                 opaths.append(p)
        #                 break
        #     od = nd.sub_dict(self.data_dict, opaths)
        #
        #     # much slower
        #     # ctask = nd.sub_dict(self.data_dict, nd.get_paths(task), compl=True)
        #     # od = nd.sample(ctask, qa[-1].shape[0], replace=True)
        #
        #     qa = np.concatenate([qa, nd.leaves_to_array(self.load_dict(od))[None,:]])
        #     ql.append('')

        return (sa, sl), (qa, ql)


def split_dict_from_folder(indir:str, ratio:float, lvl:int=None):
    return nd.random_split_level(nd.from_folder(indir), ratio, lvl)


# def sample_task(self, paths:list=None):
#     """Sample a random task.

#     Args
#     ----
#     paths: list of paths in which to sample the task. The length of a path must not exceed `self.lvl`.
#     """
#     return nd.sample(self.data_dict, self.n_way, self.lvl_task, paths)

# def sample_task_local(self):
#     """Locally sample a random task.
#     """
#     _paths = nd.get_paths(self.data_dict, self.lvl_task-1)
#     # print(_paths)
#     return self.sample_task([random.choice(_paths)])

from .tf import SequentialSampler