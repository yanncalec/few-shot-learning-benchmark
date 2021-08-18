from collections import UserDict
import functools
import operator

# An implementation of nested dictionary. Unfortuanately this is too slow to be used on large scale data, e.g. the Omniglot image database.
# Use the functions in `nested_dict.py` instead.

class NestedDict(UserDict):
    """A nested dictionary.
    """
    def __init__(self, *args):
        """Create a nested dictionary by keys
        """
        super().__init__()
        if len(args) > 0:
            self[args[0]] = args[1]

#         if type(k) in {list, tuple}:
#             if len(k) > 1:
#                 self.__init__(k[1:], v)
#                 a = self.copy()
#                 super().__init__()
#                 super().__setitem__(k[0], a)
#             else:
#                 super().__setitem__(k[0], v)
#         else:
#             if k is not None:
#                 super().__setitem__(k, v)


    def __getitem__(self, k):
        """Get an item by keys in a nested dictionary.
        """
        # Method 1:
        if type(k) in {list, tuple}:
            if len(k) > 1:
                return self[k[0]][k[1:]]
            else:
                return super().__getitem__(k[0])
        else:
            return super().__getitem__(k)
#         # Method 2:
#         if type(k) in {list, tuple}:
#             return functools.reduce(operator.__getitem__, k, self)
#         else:
#             return super().__getitem__(k)


    def __setitem__(self, k, v):
        """Set an item by keys in a nested dictionary
        """
        if type(k) in {list, tuple}:
            try:
                self[k[0]][k[1:]] = v
            except:
                super().__setitem__(k[0], NestedDict(k[1:], v) if len(k)>1 else v)
        else:
            super().__setitem__(k, v)
        self._paths = self.get_paths()

    @property
    def paths(self):
        try:
            return self._paths
        except:
            self._paths = self.get_paths()
            return self._paths

    @staticmethod
    def _get_paths(d:dict, *, lvl:int=None) -> list:
        """Recursively get all acessing paths (chains of keys) of a nested dictionary.
        """
        L = []
        if (lvl is None) or (lvl > 0):
            for k,v in d.items():
                # assert type(k) is str  # not necessary
                if type(v) is dict:  # empty dictionary must be handled
                    foo = NestedDict._get_paths(v, lvl=lvl-1 if lvl else None)
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
                        L.append(k)
                else:
                    L.append(k)
        return L

    def get_paths(self, lvl:int=None) -> list:
        """All acessing paths (chains of keys) of a nested dictionary.
        """
        L = []
        if (lvl is None) or (lvl > 0):
            for k,v in super().items():
                # assert type(k) is str  # not necessary
                if type(v) is NestedDict:  # empty dictionary must be handled
                    foo = v.get_paths(lvl-1 if lvl else None)
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

    def __contains__(self, k):
        return k in self.paths

    def __len__(self):
        """Return the number of nodes.
        """
        return len(self.paths)

    def items(self, lvl:int=None):
        _paths = self.get_paths(lvl)
        for k in _paths:
            yield k, self[k]

    def sub_dict(self, paths:list):
        """Get a sub dictionary from a set of paths.
        """
        res = NestedDict()
        for k in paths:
            assert type(k) in {list, tuple}
            res[k] = self[k]
        return res

    def random_split_level(self, ratio:float, lvl:int=None):
        _paths = self.get_paths(lvl)
        p1, p2 = random_split(_paths, int(len(_paths)*ratio))
#         return p1,p2
        return self.sub_dict(p1), self.sub_dict(p2)

    def sample(self, n:int, lvl:int=None, *, replace=False):
        _paths = self.get_paths(lvl)
        if len(_path) < n and not replace:
            raise ValueError('No enough elements.')
        random.shuffle(_paths)
        return self.sub_dict(_paths[:n])

    def keys(self):
        return self.paths
#         return super().keys()
#         print('OK')

    @staticmethod
    def from_dict(d:dict):
        res = NestedDict()

        for p in NestedDict._get_paths(d):
            v = functools.reduce(dict.__getitem__, p, d)
#             print(p, v)
            res[p] = v
        return res

    @staticmethod
    def from_folder(rootdir:str, func:Callable=None):
        """Create a nested dictionary representing the structure of a folder.
        """
        res = NestedDict()
        rootdir = regularize_filename(rootdir)
        start = len(rootdir)+1

        for path, dirs, files in os.walk(rootdir):
            folders = path[start:].split(os.sep)
            foo = NestedDict()
            for f in files:
                foo[f] = func(os.path.join(path, f)) if func else None
            res[folders] = foo

        return res