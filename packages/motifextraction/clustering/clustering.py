import os
import numpy as np
from collections import Counter
from hdbscan import HDBSCAN
from path import Path

from ..utils import FractionalCounter, load_cns


def split_affinity_by_cn(affinity, cn_dict):
    affinities = {}
    for cn, indices in cn_dict.items():
        affinities[cn] = affinity[indices[np.newaxis, :].T, indices]
    return affinities


# 1) Create a group that contains the affinity and a pointer to the parent
# 2) Run hdbscan on the group, collecting the outputs, etc
# 3) Get the child labels/indices and create child groups
# 4) Repeat 2 and 3
class HDBSCANGroup(object):
    def __init__(self,
                 affinity,
                 cns,
                 parent=None,
                 parent_label=None,
                 label='',
                 indices=None,
                 min_cluster_size_for_hdbscan=5,
                 min_cluster_size=100,
                 max_cluster_size=None,
                 min_samples=1,
                 ):
        self.affinity = affinity
        self.cns = cns
        self.parent = parent
        self._children = {}
        self.parent_label = parent_label
        self.label = label
        # Setting the cluster size parameters:
        #  min_cluster_size_for_hdbscan gets passed to the hdbscan algorithm. Either use ~ 4-5 or use min_cluster_size.
        #  min_cluster_size sets the minimum cluster size that will be accepted by this algorithm, not by hdbscan.
        #  max_cluster_size sets the maximum cluster size. It's used to determine if it's reasonable to expect only one cluster.
        #
        #  max_cluster_size should be set to be the largest counts in the CN counter.
        if max_cluster_size is None:
            max_cluster_size = Counter(self.cns).most_common(1)[0][1]
        self.max_cluster_size = max_cluster_size
        # min_cluster_size should be enough to give good statistics. Let's say 100 SRO units.
        if min_cluster_size is None:
            min_cluster_size = 100
        self.min_cluster_size = min_cluster_size
        self.min_cluster_size_for_hdbscan = min_cluster_size_for_hdbscan
        # min_samples gets passed directly to hdbscan. By default, hdbscan sets min_samples to min_cluster_size(_for_hdbscan).
        self.min_samples = min_samples

        if indices is not None:
            self._indices = indices
        else:
            if self.parent:
                self._indices = np.array([i for i, l in enumerate(self.parent.labels) if l == self.parent_label])
                self._indices = self.parent.indices[self._indices]
            else:
                self._indices = np.array(range(len(self.affinity)), dtype=np.int)

    def run(self):
        print(f"Running HDBSCAN on matrix with shape {self.affinity.shape}...")
        self.db = self._run_hdbscan(self.affinity,
                                    self.min_cluster_size_for_hdbscan,
                                    self.min_cluster_size,
                                    self.max_cluster_size)
        self.label_counter = Counter(self.db.labels_)
        print(self.label_counter)
        self.biggest_cluster_index = self.label_counter.most_common(1)[0][0]
        if self.biggest_cluster_index == -1 and len(self.label_counter) > 1:
            self.biggest_cluster_index = self.label_counter.most_common(2)[1][0]

    @staticmethod
    def _run_hdbscan(affinity: np.ndarray, min_cluster_size_for_hdbscan: int, min_cluster_size: int, max_cluster_size: int):
        assert affinity.shape[0] == affinity.shape[1]
        if affinity.shape[0] > max_cluster_size:
            allow_single_cluster = False
        else:
            allow_single_cluster = True
        db = HDBSCAN(metric='precomputed',
                     min_cluster_size=min_cluster_size_for_hdbscan,
                     min_samples=1,
                     allow_single_cluster=allow_single_cluster)
        db.fit(affinity)
        return db

    def run_recursive_hdbscan(self):
        self.run()
        for label, count in self.label_counter.most_common():
            if count > self.min_cluster_size and len(self.label_counter) > 1:
                child_affinity = self.make_child_affinity(label)
                child = HDBSCANGroup(affinity=child_affinity,
                                     cns=self.cns,
                                     parent=self,
                                     min_cluster_size_for_hdbscan=self.min_cluster_size_for_hdbscan,
                                     min_cluster_size=self.min_cluster_size,
                                     max_cluster_size=self.max_cluster_size,
                                     min_samples=self.min_samples,
                                     parent_label=label,
                                     label=self.label + str(label)
                                     )
                self._children[label] = child
                child.run_recursive_hdbscan()

    @property
    def indices(self):
        return self._indices

    def get_params(self):
        return self.db.get_params()

    @property
    def labels(self):
        return self.db.labels_

    def make_child_affinity(self, label):
        indices = np.array([i for i, l in enumerate(self.labels) if l == label])
        return self.affinity[indices[np.newaxis, :].T, indices]

    @property
    def center(self):
        index = np.argmin(np.mean(self.affinity, axis=0))
        index = self.indices[index]
        return index

    def recursive_print(self, cns: np.ndarray, save_dir: Path):
        # `cns` is a numpy array of all the CNs of the SRO units
        if save_dir is not None and not os.path.exists(save_dir):
            os.makedirs(save_dir)
        print_this = True
        if self.parent is not None:
            # If the same center, this child just has less points in it so I want to prioritize the parent because it
            #  has more points.
            if self.parent.center == self.center:
                print_this = False
        #else:
        #    print_this = False

        if print_this:
            print(f"Input affinity size: {len(self.affinity)}  Center: {self.center}")
            print(f"Label: {self.label}")
            if cns is not None:
                _cns = cns[self.indices]
                print(f"CNs in this cluster: {FractionalCounter(_cns)}")
            print(f"HDBSCAN results: {self.label_counter}")
            print('')
            if save_dir is not None:
                np.savetxt(f"{save_dir}/{self.label}.txt", self.indices, header=f"{self.center} {self.label}")
        for label, child in self._children.items():
            child.recursive_print(cns, save_dir=save_dir)
