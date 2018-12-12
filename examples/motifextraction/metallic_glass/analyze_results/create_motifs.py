import sys
from path import Path

from motifextraction.analyze_results import create_motif


def main():
    data_path = Path(sys.argv[1])
    cluster_path = Path('../data/clusters/')
    labels_path = data_path / 'refined_indices/'
    aligned_data_path = Path('../data/results/')
    for label in labels_path.glob("*"):
        create_motif(cluster_path=cluster_path,
                     label_path=label,
                     results_path=aligned_data_path,
                     output_data_path=Path("../data/"),
                     )


if __name__ == "__main__":
    main()
