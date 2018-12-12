from path import Path
from natsort import natsorted


def find_failed():
    nclusters = len(Path("../data/clusters/").glob("*.xyz"))
    needed_alignments = set([i for i in range(nclusters)])
    alignments = set([int(fn.name.split('.')[0]) for fn in Path("../data/results/").glob("*.xyz.json")])
    errors = set([int(fn.name.split('_')[0]) for fn in Path("../data/errors/").glob("*.npy")])
    missing_alignments = needed_alignments - alignments
    missing_errors = (needed_alignments - errors) - missing_alignments
    if len(missing_alignments) < 150:
        for i in natsorted(missing_alignments):
            cn = int(open("../data/clusters/{}.xyz".format(i)).readline().strip()) - 1
            print("{}  has CN {}".format(i, cn))
    print("Total alignments missing: {}".format(len(missing_alignments)))
    if len(missing_errors) < 150:
        for i in natsorted(missing_errors):
            cn = int(open("../data/clusters/{}.xyz".format(i)).readline().strip()) - 1
            print("{}  has CN {}".format(i, cn))
    print("Additional errors missing: {}".format(len(missing_errors)))
    return missing_alignments, missing_errors


if __name__ == '__main__':
    import subprocess
    #subprocess.run("./move_results.sh")
    find_failed()
