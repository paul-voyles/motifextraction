import sys
from path import Path
from natsort import natsorted

from motifextraction.alignment import run_alignments


def align_all_to(cluster_number):
    save_dir = Path('../data/results')

    clusters_path = Path('../data/clusters')
    xyz_files = natsorted(clusters_path.glob("*.xyz"))

    model_files = xyz_files
    target_files = [xyz_files[cluster_number]]
    print(target_files)
    assert len(model_files) > len(target_files)

    run_alignments(target_files=target_files, model_files=model_files, save_dir=save_dir)


def main():
    cluster_number = int(sys.argv[1])
    align_all_to(cluster_number)


if __name__ == '__main__':
    main()
