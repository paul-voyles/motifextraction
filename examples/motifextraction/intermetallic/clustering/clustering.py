import sys
from path import Path
from collections import defaultdict

from motifextraction import load_cns, load_affinity
from motifextraction.clustering import HDBSCANGroup, split_affinity_by_cn


def main():
    affinity_path = Path(sys.argv[1])
    affinity = load_affinity(affinity_path)

    cns = load_cns(Path("../data/clusters/"))
    nclusters = len(cns)

    #min_cluster_size_for_hdbscan = int(round(0.01 * nclusters))  # 1% of the total number of clusters
    min_cluster_size_for_hdbscan = 5
    print(f"Min cluster size for HDBSCAN: {min_cluster_size_for_hdbscan}  (Total # of clusters: {nclusters})")
    save_dir = affinity_path.parent / "indices/"

    split_by_cn = False
    if not split_by_cn:
        g0 = HDBSCANGroup(affinity,
                          cns=cns,
                          parent=None,
                          parent_label=None,
                          label='_',
                          indices=None,
                          min_cluster_size_for_hdbscan=min_cluster_size_for_hdbscan,
                          min_cluster_size=10,
                          max_cluster_size=None,
                          min_samples=1,
                          )
        g0.run_recursive_hdbscan()
        print('')
        g0.recursive_print(cns, save_dir=save_dir)
    else:
        cn_dict = defaultdict(list)
        for i, cn in enumerate(cns):
            cn_dict[cn].append(i)
        affinities = split_affinity_by_cn(affinity, cn_dict)
        for cn, affinity in affinities.items():
            print("CN: {}".format(cn))
            g0 = HDBSCANGroup(affinity,
                              cns=cns,
                              parent=None,
                              parent_label=None,
                              label=f'CN{cn}_',
                              indices=cn_dict[cn],
                              min_cluster_size_for_hdbscan=4,
                              min_cluster_size=100,
                              max_cluster_size=None,
                              min_samples=1,
                              )
            g0.run_recursive_hdbscan()
            print('')
            g0.recursive_print(cns, save_dir=save_dir)


if __name__ == '__main__':
    main()
