from path import Path
import numpy as np
from collections import Counter


def load_label(fn):
    return np.loadtxt(fn, skiprows=1).astype(int)


def load_labels(direc: Path):
    labels = []
    for fn in direc.glob("*"):
        stem = fn.stem
        #assert stem.startswith('_')
        assert stem.startswith('CN')
        _index = stem.index('_')
        stem = stem[_index+1:]
        all_negative_ones = (len(stem.replace("-1", "")) == 0)
        labels.append(load_label(fn))
    return labels


def nclusters(label):
    return len(label)


def get_cn_composition(label, cluster_cns):
    return Counter(cluster_cns[label])


def refine_indices(affinity, min_cluster_size, clusters, cluster_cns, labels_path: Path, new_labels_path: Path):
    if not new_labels_path.exists():
        new_labels_path.makedirs()

    labels = load_labels(labels_path)
    increment = 0
    for label in labels:
        cn_composition = get_cn_composition(label, cluster_cns)
        for cn, count in cn_composition.most_common():
            if count > min_cluster_size:
                cns = cluster_cns[label]
                l = label[np.where(cns == cn)]
                a = affinity[l[np.newaxis, :].T, l]
                am = np.mean(a, axis=1)
                center = l[np.argmin(am)]
                print(f"# {len(l)} {cn} {center}")
                with open(f"{new_labels_path}/{increment}.txt", 'w') as f:
                    f.write(f"# {len(l)} {cn} {center}\n")
                    for x in l:
                        f.write(f"{x}\n")
                increment += 1
        print(sum(cn_composition.values()), cn_composition)
