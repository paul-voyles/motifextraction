import sys
import numpy as np
from typing import Iterable

from ppm3d import Cluster
from motifextraction.utils import load_affinity, load_clusters
from motifextraction.clustering import refine_indices
from path import Path


def get_cluster_cns(clusters: Iterable[Cluster]):
    return np.array([c.CN for c in clusters])


def main():
    cluster_path = Path('../data/clusters/')
    clusters = load_clusters(cluster_path)
    cluster_cns = get_cluster_cns(clusters)

    affinity_path = Path(sys.argv[1])
    affinity = load_affinity(affinity_path)

    sigma_path = affinity_path.parent
    labels_path = sigma_path / 'indices/'
    new_labels_path = sigma_path / 'refined_indices/'

    refine_indices(min_cluster_size=10,
                   affinity=affinity,
                   clusters=clusters,
                   cluster_cns=cluster_cns,
                   labels_path=labels_path,
                   new_labels_path=new_labels_path)


if __name__ == "__main__":
    main()
