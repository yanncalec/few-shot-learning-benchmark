"""Functions for manipulation of the nested dictionaries.
"""

import os
import functools
import operator
from typing import Callable
from abc import ABC
import random

import numpy as np

# from skimage.io import imread
# from skimage.transform import resize
# from imageio import imread
from PIL import Image

import logging
Logger = logging.getLogger('fsl.utils.nested_dict')

from . import data


def random_split(a:list, s:int):
    """Randomly split a list `a` into two parts with the first one having length `s`.
    """
    assert 0 <= s <= len(a)
    idx = list(range(len(a))); random.shuffle(idx)
#     a1 = [a[idx[i]] for i in range(0, min(s,len(a)))]
#     a2 = [a[idx[i]] for i in range(max(0,s),len(a))]
    a1 = [a[idx[i]] for i in range(0, s)]
    a2 = [a[idx[i]] for i in range(s,len(a))]
    return a1, a2
    # # equivalent to
    # aa = a.copy(); random.shuffle(aa)
    # return aa[:s]; aa[s:]


def nested_dict(k, v):
    """Create a nested dictionary by keys.

    Example
    -------
    `nested_dict(['a','b','c'], 0)` returns `{'a':{'b':{'c':0}}}`.
    """
    if type(k) in {list, tuple}:
        d = {k[-1]: v}
        for q in k[:-1][::-1]:
            d = {q: d}
        return d
    else:
        return {k: v}


def getitem(d:dict, k:list):
    """Get an item by keys in a nested dictionary.

    Example
    -------
    For the nested dictionary `{'a':{'b':{'c':0}}}`, query with the key `['a','b','c']` returns 0;
    and query with the key `['a','b']` returns `{'c':0}`.
    """
    # retrieve from a nested dictionary
    # possible to use dict.get() or operator.getitem()
    return functools.reduce(dict.__getitem__, k, d)


def setitem(d:dict, k, v):
    """Set an item by keys in a nested dictionary.

    Example: for the nested dictionary `d={'a':{'b':{'c':0}}}`, `k=['a','b','c']` and `v=1` set the dictionary to `d={'a':{'b':{'c':1}}}`.
    """
    assert type(d) is dict

    if type(k) in {list, tuple}:
        try:
#             print(d,k,v)
            setitem(d[k[0]], k[1:], v)
        except:
            d[k[0]] = nested_dict(k[1:], v) if len(k) > 1 else v
    else:
        d[k] = v

# def setitem(d, k, v):
#     # works only if `d` has the key `k[:-1`
#     p = functools.reduce(dict.__getitem__, k[:-1], d)
#     p[k[-1]] = v


def _get_paths_pos(d:dict, lvl:int=None) -> list:
    L = []
    if (lvl is None) or (type(lvl) is int and lvl >= 0):
        for k,v in d.items():
            # print(lvl,k,v)
            # assert type(k) is str  # not necessary
            if type(v) is dict:  # empty dictionary must be handled
                foo = _get_paths_pos(v, None if lvl is None else lvl-1)
                if foo:
                    # L += [k+sep+t for t in foo]  # if a separator is used, working only for string keys
                    poo = []
                    for t in foo:
                        if type(t) is list:
                            poo.append([k, *t])
                        else:
                            poo.append([k, t])
                    L += poo
                    # L += [[k, *t] for t in foo]  # trouble if t is not a list
                else:
                    L.append([k])
            else:
                L.append([k])
    return L


def _get_paths_neg(d:dict, lvl:int):
    assert lvl < 0
    foo = [f[:lvl] for f in _get_paths_pos(d, None)]
    paths = []
    for f in foo:
        if f not in paths:
            paths.append(f)
    return paths


def get_paths(d:dict, lvl:int=None):
    """Get all acessing paths (chains of keys) of a nested dictionary.

    Example: for `d={'a':{'b':{'c':0, 'd':1}}}`, `lvl=None` returns `[['a','b','c'], ['a','b','d']]`; while `lvl=0` returns `[['a']]`, and `lvl=-1` returns `[['a','b']]`.
    """
    if lvl is None:
        return _get_paths_pos(d, lvl)
    else:
        if lvl>=0:
            return _get_paths_pos(d, lvl)
        else:
            return _get_paths_neg(d, lvl)
        # raise ValueError('Level must be an integer or None.')


def sub_dict(d:dict, paths:list, *, compl=False):
    """Get a sub dictionary from a set of paths.

    If `compl` is True then take the complementary of the given paths.

    Example: for `d={'a':{'b':{'c':0, 'd':1}}, 'e':{'f':2, 'g':3}}`, `path=[['a'],['e','f']]` returns `{'a':{'b':{'c':0, 'd':1}}, 'e':{'f':2}}`. When `compl=True` then `path=[['a']]` returns `{'e':{'f':2, 'g':3}}`.
    """
#     k = keys[0]
#     assert type(k) in {list, tuple}
#     res = nested_dict(k, fsl.utils.data.get_item(d, k))
    res = {}
    if compl:
        pp = []
        for p in get_paths(d):
            for q in paths:
                if q == p[:len(q)]:
                    break
            else:
                pp.append(p)
    else:
        pp = paths

    for k in pp:
        # assert type(k) in {list, tuple}
        setitem(res, k, getitem(d, k))
    return res


def random_split_level(d:dict, ratio:float, lvl:int=None, *, index:int=None):
    """Randomly split a nested dictionary at a given level.
    """
    _paths = get_paths(d, lvl)
    if index is None:
        p1, p2 = random_split(_paths, int(len(_paths)*ratio))
    else:  # if split index is provided
        p1, p2 = random_split(_paths, index)
    # return p1,p2
    return sub_dict(d, p1), sub_dict(d, p2)


def random_split_level_local(d:dict, ratio:float, lvl:int=None, *, index:int=None):
    """Locally split a nested dictionary at a given level.
    """
    # assert lvl > 0

    d1, d2 = {}, {}
    if lvl is None:
        paths = _get_paths_neg(d, -1)
    else:
        paths = get_paths(d, lvl-1)

    for p in paths:
        s1, s2 = random_split_level(getitem(d, p), ratio, 0, index=index)
        setitem(d1, p, s1)
        setitem(d2, p, s2)
    return d1, d2


def sample(d:dict, n:int, lvl:int=None, *, replace=False):
    """Generate random samples from a nested dictionary at a given level.
    """
    _paths = get_paths(d, lvl)
    if replace:
        p = random.choices(_paths, k=n)  # with replacement
        # assert len(p) == n
        return sub_dict(d, p)
    else:
        if len(_paths) < n:
            raise ValueError('No enough elements.')
        random.shuffle(_paths)
        return sub_dict(d, _paths[:n])


def sample_local(d:dict, n:int, lvl:int=None, *, replace=False):
    """Locally generate random samples from a nested dictionary at a given level.
    """
    res = {}
    if lvl is None:
        paths = _get_paths_neg(d, -1)
    else:
        paths = get_paths(d, lvl-1)

    for p in paths:
        s = sample(getitem(d, p), n, 0, replace=replace)
        setitem(res, p, s)
    # for p in get_paths(d, lvl):
    #     s = sample(getitem(d, p[:-1]), n, 0, replace=replace)
    #     setitem(res, p[:-1], s)
    return res


# Sampling of the support and query set
def support_query_sampling(task:dict, ns:int, nq:int=None, lvl:int=None, *, split=False):
    """
    """
    if split:
        sd, qd = random_split_level_local(task, 0, lvl, index=ns)
        if nq:
            qd = sample_local(qd, nq, lvl)
    else:
        sd = sample_local(task, ns, lvl, replace=False)  # dict of the support set
        qd = sample_local(task, nq, lvl, replace=False) if nq else task.copy() # dict of the query set

    return sd, qd


def from_folder(rootdir:str, func:Callable=None, filt:Callable=None) -> dict:
    """Create a nested dictionary representing the structure of a folder.

    Arguments
    ---------
    rootdir: path of a root folder
    func: function processing the content of a file, optional
    filt: boolean function filtering the input file names, optional

    References
    ----------
    - https://code.activestate.com/recipes/577879-create-a-nested-dictionary-from-oswalk/
    - https://stackoverflow.com/questions/28225552/is-there-a-recursive-version-of-the-dict-get-built-in
    - https://lerner.co.il/2014/05/11/creating-python-dictionaries-reduce/
    """

    res = {}
    # count = 0  # counter of the number of leaves
    rootdir = data.regularize_filename(rootdir)
    # rootdir = rootdir.rstrip(os.sep)  # remove the trailing '/'
    start = rootdir.rfind(os.sep) + 1

    if filt is None:
        filt = lambda x: True

    for path, _, files in os.walk(rootdir):
        folders = path[start:].split(os.sep)

        subdir = {f: func(os.path.join(path, f)) if func else None for f in files if filt(f)}
        # # equivalent to:
        # subdir = dict.fromkeys(files)
        # for f in files:  # load image files if found
        #     if func:
        #         subdir[f] = func(os.path.join(path, f))
        #     else:
        #         subdir[f] = None

        # count += len(files)

        if subdir:  # do not add node, in case of empty dictionary
            setitem(res, folders, subdir)

    # pop out the root level item
    return res.popitem()[1] if res else res


def count_leaves(d:dict) -> int:
    """Count the number of leaves in a nested dictionary.
    """
    n = 0
    for k,v in d.items():
        if type(v) is dict:
            n += count_leaves(v)
        else:
            n += 1
    return n

# count_leaves(Data)/20  == 1623

def count_nodes(d:dict) -> int:
    """Count the number of nodes (leaves included) in a nested dictionary.
    """
    n = len(d.keys())
    for k,v in d.items():
        if type(v) is dict:
            n += count_nodes(v)
    return n


def count_leafnodes(d:dict) -> int:
    """Count the number of nodes with leaves in a nested dictionary.
    """
    n = 0
    if d:
        m = 0
        for k,v in d.items():
            if type(v) is dict:
                m += count_leafnodes(v)
        return n + max(m,1)
    return n


