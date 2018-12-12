import numpy as np
from collections import Counter
from natsort import natsorted
from path import Path

from ppm3d import Cluster


def load_clusters(direc: Path):
    files = natsorted(direc.glob("*.xyz"))
    return [Cluster(filename=fn) for fn in files]


def get_norm_factors(filename) -> dict:
    affinity = np.load(filename)
    affinity = np.minimum(affinity, affinity.T)
    good_affinity = affinity[np.isfinite(affinity) & ~np.isnan(affinity)]
    affinity[np.where(affinity > np.mean(good_affinity) * 5)] = np.inf
    good_affinity = affinity[np.isfinite(affinity) & ~np.isnan(affinity)]
    mean = np.mean(good_affinity)
    max_ = np.amax(good_affinity)
    affinity[np.where(np.isinf(affinity))] = max_ * 3.0
    affinity[np.where(np.isnan(affinity))] = max_ * 3.0
    norm_results = {'set_to_inf_before_dividing': max_ * 3.0, 'divide_by': mean}
    return norm_results


def load_affinity(filename, normalize=True) -> np.ndarray:
    print("Loading {} affinity...".format(filename))
    affinity = np.load(filename)
    if not normalize:
        return affinity
    #print("Making symmetric...")
    affinity = np.minimum(affinity, affinity.T)
    #print("Finished making symmetric!")
    #print("Normalizing...")
    good_affinity = affinity[np.isfinite(affinity) & ~np.isnan(affinity)]
    print("Setting all values above {} to np.inf".format(np.mean(good_affinity) * 5))
    affinity[np.where(affinity > np.mean(good_affinity) * 5)] = np.inf
    good_affinity = affinity[np.isfinite(affinity) & ~np.isnan(affinity)]
    mean = np.mean(good_affinity)
    max_ = np.amax(good_affinity)
    #print("Mean before normalizing: {}".format(mean))
    #print("Max  before normalizing: {}".format(max_))
    affinity[np.where(np.isinf(affinity))] = max_ * 3.0
    affinity[np.where(np.isnan(affinity))] = max_ * 3.0
    print("Setting inf and nan values to {}".format(max_ * 3.0))
    min_val = np.amin(affinity)
    assert np.isclose(min_val, 0)
    #print("Subtracting {}".format(min_val))
    #affinity = affinity - min_val
    print("Dividing by {}".format(mean))
    affinity = affinity / mean
    norm_results = {'set_to_inf_before_dividing': max_ * 3.0, 'divide_by': mean}
    print(norm_results)
    #print("New mean: {}  (should be a bit larger than 1.0)".format(np.mean(affinity)))
    #print("New max:  {}".format(np.amax(affinity)))
    return affinity


def load_cns(direc: Path):
    print("Loading cns...")
    nclusters = len([fn for fn in direc.glob("*.xyz")])
    cns = np.zeros((nclusters,))
    cns.fill(np.nan)
    clusters = [direc / f'{i}.xyz' for i in range(nclusters)]
    for i, xyz in enumerate(clusters):
        with open(xyz) as f:
            natoms = int(f.readline())
        cns[i] = natoms - 1
    assert (~np.isnan(cns)).all()
    return cns


class FractionalCounter(Counter):
    def __init__(self, *args, precision=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.precision = precision

    def _total(self):
        total = 0
        for key in self:
            total += Counter.__getitem__(self, key)
        return total

    def __getitem__(self, key):
        return Counter.__getitem__(self, key) / self._total()

    def __str__(self):
        if self.precision is None:
            return str({key: self[key] for key in self})
        else:
            return str({key: f'{round(self[key]*100, self.precision)}%' for key in self})

    def __repr__(self):
        return self.__str__()
