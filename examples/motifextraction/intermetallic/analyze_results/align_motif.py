import sys
from path import Path
from natsort import natsorted

from motifextraction.alignment import run_alignments


def align_all_to(motif_path):
    save_dir = Path('../data/motif_results')

    clusters_path = Path('../data/clusters')
    xyz_files = natsorted(clusters_path.glob("*.xyz"))

    model_files = xyz_files
    target_files = [motif_path]
    print(target_files)
    assert len(model_files) > len(target_files)

    run_alignments(target_files=target_files, model_files=model_files, save_dir=save_dir)


def main():
    motif_path = Path(sys.argv[1])
    align_all_to(motif_path=motif_path)


if __name__ == '__main__':
    main()
