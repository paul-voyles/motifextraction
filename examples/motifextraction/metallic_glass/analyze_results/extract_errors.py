import sys
from path import Path

from motifextraction.alignment import extract_errors


def main():
    motif_name = Path(sys.argv[1]).stem  # e.g. "../data/averaged/averaged_0.xyz" => "averaged_0"
    cutoff = float(sys.argv[2])
    results_path = Path("../data/motif_results/")
    results_fn = results_path / (motif_name + ".xyz.json")
    extract_errors(cluster_number=motif_name,
                   results_fn=results_fn,
                   save_dir=Path("../data/motif_errors/"),
                   cluster_dir=Path("../data/clusters/"),
                   cutoff=cutoff,
                   target_path=Path("../data/averaged/"))


if __name__ == '__main__':
    main()
