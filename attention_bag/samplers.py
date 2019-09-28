from __future__ import absolute_import
import torch

import numpy as np
from collections import defaultdict
from torch.utils.data import Sampler


class RandomIdentitySampler(Sampler):
    """
    randomly sample N identities
    randomly sample k = 2 instances(image) for one identities
    """
    def __init__(self, data_source, num_instances=4):
        self.data_source = data_source
        self.num_instances = num_instances
        self.index_dic = defaultdict(list)

        for index, (_, pid ,_) in enumerate(data_source):
            self.index_dic[pid].append(index)
        self.pids = list(self.index_dic.keys())
        self.num_identities = len(self.pids)

    def __iter__(self):
        indices = torch.randperm(self.num_identities)
        ret = []
        for i in indices:
            pid = self.pids[i]
            t = self.index_dic[pid]
            t = np.random.choice(t, size=self.num_instances)
            ret.extend(t)
        return iter(ret)

    def __len__(self):
        return self.num_instances * self.num_identities

