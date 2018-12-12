import os
import numpy as np
from scipy.optimize import curve_fit
from path import Path
from typing import List, Iterable

from ..utils import load_affinity


def _load_info_affinities(affinities, cluster_number, fn):
    if cluster_number % 10 == 0:
        print("Loading {0}...".format(fn))
    try:
        affinities[cluster_number, :, :] = np.load(fn)
    except IOError:
        print("FAILED", fn)


def create_affinities(nclusters: int, error_files: Iterable[Path], affinity_paths: List[str], cluster_cns: Iterable[int], affinities_path: Path):
    if not os.path.exists(affinity_paths[0]):
        affinities = np.zeros((nclusters, nclusters, len(affinity_paths)), dtype=np.float)
        affinities.fill(np.inf)

        for cluster_number, fn in enumerate(error_files):
            _load_info_affinities(affinities, cluster_number, fn)
        print("Finished loading!!!")

        # Save the affinities
        if not os.path.exists(affinities_path):
            os.makedirs(affinities_path)
        for i, affinity_name in enumerate(affinity_paths):
            affinity = affinities[:, :, i]
            np.save(affinity_name, affinity)
        del affinities, affinity  # Clear the memory

    # Create the combined affinity
    # Reload the affinities so they get normalized properly
    if not os.path.exists(f'{affinities_path}/combined_affinity.npy'):
        combined = load_affinity(affinity_paths[0])
        for affinity_name in affinity_paths[1:]:
            combined += load_affinity(affinity_name)
        np.save(f'{affinities_path}/combined_affinity.npy', combined)


# find the longest strings of 0s in the histogram and truncate it there, then fit to a gaussian
# we will use this number to decide what to scale the CN affinity by before adding it to this combined affinity
def get_truncated_histogram(center, hist, bins):
    longest = 0
    start = 0
    length = 0
    for i, x in enumerate(hist):
        if x == 0:
            length += 1
        else:
            length = 0
        if length > longest:
            longest = length
            start = i - length + 1

    center = (bins[:-1] + bins[1:]) / 2
    return center[:start], hist[:start]


def fit_gaussian(center, hist, plot=False):
    x,y = center.copy(), hist.copy()

    mean = x[np.argmax(y)]
    sigma = (x[np.argmax(y)] - x[0])/2.0

    def gaus(x, a, x0, sigma):
        return a*np.exp(-(x-x0)**2/(2*sigma**2))

    popt,pcov = curve_fit(gaus, x, y, p0=[np.amax(y),mean,sigma])

    if plot is True:
        import matplotlib.pyplot as plt
        plt.plot(x,y,'b+:',label='data')
        plt.plot(x,gaus(x,*popt),'ro:',label='fit')
        plt.legend()
        plt.show()
    return popt


def fit_truncated_gaussian(center, hist, bins, plot=False):
    truncated_center, truncated_hist = get_truncated_histogram(center, hist, bins)
    x,y = truncated_center, truncated_hist

    mean = x[np.argmax(y)]
    sigma = (x[np.argmax(y)] - x[0])/2.0

    def gaus(x,a,x0,sigma):
        return a*np.exp(-(x-x0)**2/(2*sigma**2))

    popt,pcov = curve_fit(gaus, x, y, p0=[np.amax(y),mean,sigma])

    if plot is True:
        import matplotlib.pyplot as plt
        plt.plot(x,y,'b+:',label='data')
        plt.plot(x,gaus(x,*popt),'ro:',label='fit')
        plt.legend()
        plt.show()
    return popt
