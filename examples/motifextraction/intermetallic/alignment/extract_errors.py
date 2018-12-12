import sys
from path import Path

from motifextraction.alignment import extract_errors


def main():
    results_fn = Path(sys.argv[1])
    cutoff = float(sys.argv[2])
    cluster_number = int(results_fn.name.split('.')[0])
    extract_errors(cluster_number=cluster_number,
                   results_fn=results_fn,
                   save_dir=Path("../data/errors/"),
                   cluster_dir=Path("../data/clusters/"),
                   cutoff=cutoff)


if __name__ == '__main__':
    main()
