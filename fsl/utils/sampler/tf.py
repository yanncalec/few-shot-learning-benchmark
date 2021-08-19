import numpy as np
import random
import math
from tensorflow import keras

from . import MetaSampler

class SequentialSampler(MetaSampler, keras.utils.Sequence):
# class SequentialSampler(MetaSampler):
    # def __init__(self, *args, **kwargs):
    #     super().__init__(*args, **kwargs)

    def __getitem__(self, idx):
        (sa, sl), (qa, ql) = self.sample_support_query(self.get_task(idx), split=self.split)
        n_way, k_shot = sa.shape[:2]
        _, q_shot = qa.shape[:2]
        yt = np.tile(np.arange(n_way), (q_shot,1)).T.flatten()
        # print(qa.shape, sa.shape)
        # print(sa[0,0])
        return [qa, sa], yt

    def generator(self):
        for x in self:
            yield x

    def on_epoch_end(self):
        self.shuffle()