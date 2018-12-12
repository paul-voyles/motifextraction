import numpy as np
from collections import Counter, defaultdict
from itertools import combinations
import scipy.spatial, scipy.stats, scipy.misc
import ppm3d
from ppm3d import Cluster, render_alignment_data
from motifextraction import load_cns
from path import Path
from natsort import natsorted
import json

import multiprocessing


def align_one(p1, p2, norm_factors, cutoff):
    out = ppm3d.align(model=p1.filename,
                      target=p2.filename,
                      normalize_edges=True,
                      check_inverse=True,
                      use_combinations=False,
                      max_alignments=1)
    aligned = render_alignment_data([out], prepend_path=".")[0]
    rcutoff = cutoff / (0.5*aligned.model_scale + 0.5*aligned.target_scale)
    l2, l1, linf, angular = aligned.L2Norm(), aligned.L1Norm(), aligned.LinfNorm(), aligned.angular_variation(rcutoff)
    if l2 > norm_factors['L2']['set_to_inf_before_dividing']:
        l2 = np.inf
    l2 /= norm_factors['L2']['divide_by']
    if l1 > norm_factors['L1']['set_to_inf_before_dividing']:
        l1 = np.inf
    l1 /= norm_factors['L1']['divide_by']
    if linf > norm_factors['Linf']['set_to_inf_before_dividing']:
        linf = np.inf
    linf /= norm_factors['Linf']['divide_by']
    if angular > norm_factors['angular']['set_to_inf_before_dividing']:
        angular = np.inf
    angular /= norm_factors['angular']['divide_by']
    errors = np.array([l2, l1, linf, angular])
    errors[np.where(np.isinf(errors))] = np.nan
    error = scipy.stats.mstats.gmean(errors)
    print("{:14f} {} {}".format(error, p1.filename, p2.filename))
    return error


def align_one_(args):
    p1, p2, norm_factors, cutoff = args
    return align_one(p1, p2, norm_factors, cutoff)


def align_motifs(motifs, norm_factors, cutoff):
    # Align the motifs to one another
    combs = list(combinations(motifs, 2))
    pool = multiprocessing.Pool(20)
    inputs = [(p1, p2, norm_factors, cutoff) for p1, p2 in combs]
    results = pool.map(align_one_, inputs)

    motif2motif_errors = {}
    for (p1,p2), error in zip(combs, results):
        motif2motif_errors[(str(p1.filename), str(p2.filename))] = error
        motif2motif_errors[(str(p2.filename), str(p1.filename))] = error
    #for p1, p2 in combs:
    #    error = align_one(p1, p2, norm_factors, cutoff)
    #    print("{:14f} {} {}".format(error, p1.filename, p2.filename))
    #    motif2motif_errors[(p1.filename, p2.filename)] = error
    #    motif2motif_errors[(p2.filename, p1.filename)] = error
    return motif2motif_errors


def choose(errors_path: Path, norm_factors: dict):
    cluster_path = Path("../data/clusters")
    nclusters = len(cluster_path.glob("*.xyz"))

    print("Loading motifs...")
    motifs = natsorted(Path("../data/averaged/").glob("*.xyz"))
    motifs = [Cluster(filename=p) for p in motifs]
    print("Total prototypes: {}".format(len(motifs)))
    print("Number of prototypes by CN: {}".format(sorted(Counter(motif.CN for motif in motifs).items())))

    # Remove motifs that are obviously bad (e.g. two atoms too close together)
    to_remove = []
    mindists = []
    for motif in motifs:
        mindist = np.amin(scipy.spatial.distance.pdist(motif.positions))
        mindists.append(mindist)
        if mindist < 2.3*0.9:  # cutoff taken from g(r)
            to_remove.append(motif)
            #print(motif.filename, mindist)
            raise RuntimeError("This shouldn't happen. Two atoms in the motif are too close together. Check the combined model.")
    motifs = [m for m in motifs if m not in to_remove]
    print(f"Found {len(motifs)} motifs with reasonable bond distances.")

    # Get the alignment errors for each motif
    for motif in motifs:
        tail = Path(motif.filename).name[:-4]
        try:
            #print(errors_path.parent / f"motif_errors/{tail}_errors.npy")
            errors = np.load(errors_path.parent / f"motif_errors/{tail}_errors.npy")
            assert errors.shape[1] == 4
            # Apply norm factors then gmean
            l2, l1, linf, angular = errors[:, 0], errors[:, 1], errors[:, 2], errors[:, 3]
            l2[np.where(l2 > norm_factors['L2']['set_to_inf_before_dividing'])] = np.inf
            l2 /= norm_factors['L2']['divide_by']
            l1[np.where(l1 > norm_factors['L1']['set_to_inf_before_dividing'])] = np.inf
            l1 /= norm_factors['L1']['divide_by']
            linf[np.where(linf > norm_factors['Linf']['set_to_inf_before_dividing'])] = np.inf
            linf /= norm_factors['Linf']['divide_by']
            angular[np.where(angular > norm_factors['angular']['set_to_inf_before_dividing'])] = np.inf
            angular /= norm_factors['angular']['divide_by']
            errors = np.array([l2, l1, linf, angular]).T
            errors[np.where(np.isinf(errors))] = np.nan
            assert errors.shape[1] == 4
            errors = scipy.stats.mstats.gmean(errors, axis=1)
            motif.errors = errors
        except IOError:
            print(f"Failed on {tail}_errors.npy")
            continue

    run_motif2motif_alignments = False
    motif2motif_errors_path = Path("../data/motif2motif_errors.json")
    if run_motif2motif_alignments and not motif2motif_errors_path.exists():
        motif2motif_errors = align_motifs(motifs, norm_factors, cutoff=3.45)
        with open(motif2motif_errors_path, "w") as f:
            _motif2motif_errors = {f'{f1} + {f2}': error for (f1, f2), error in motif2motif_errors.items()}
            json.dump(_motif2motif_errors, f)

    # The similarity values change depending on CN so we need a different threshold for each.
    # To figure them out, set the threshold to 0 for all of them and plot the resulting print outs
    #  for the maximum similarity values. Choose the thresholds from that plot.
    thresholds = {9:  0.0,
                  10: 0.0,
                  11: 0.0,
                  12: 0.0,
                  13: 0.0,
                  14: 0.0,
                  15: 0.0,
    }
    cns = load_cns(Path("../data/clusters"))
    motifs_by_CN = separate_motifs_by_CN(motifs)
    for CN, ms in motifs_by_CN.items():
        ms = run_removal(ms, threshold=0.0, cns=cns, norm_factors=norm_factors)  # 0.45 if using PRs as similarity
        #ms = run_removal(ms, threshold=thresholds[CN], cns=cns, norm_factors=norm_factors)  # 0.45 if using PRs as similarity
        motifs_by_CN[CN] = ms
    motifs = [ms for ms in motifs_by_CN.values()]
    motifs = [motif for ms in motifs for motif in ms]

    print("After removing similar prototypes, {} are left.".format(len(motifs)))
    print("They are:")
    for p in motifs:
        print("{} with CN {}".format(p.filename, p.CN))
    print("Number of prototypes by CN: {}".format(sorted(Counter(p.CN for p in motifs).items())))


def get_similarities(motifs, norm_factors, cns):
    pearsons_Rs = {}
    for p1, p2 in combinations(motifs, 2):
        assert p1.CN == p2.CN
        nans = np.logical_or(np.isnan(p1.errors), np.isnan(p2.errors))
        a = np.where(~nans & (cns == p1.CN))
        pr = scipy.stats.pearsonr(p1.errors[a], p2.errors[a])[0]
        pearsons_Rs[(p1.filename, p2.filename)] = pr
        pearsons_Rs[(p2.filename, p1.filename)] = pr
    return pearsons_Rs


#def get_similarities(motifs, norm_factors, cns):
#    # Load motif2motif_errors
#    motif2motif_errors_path = Path("../data/motif2motif_errors.json")
#    if motif2motif_errors_path.exists():
#        with open(motif2motif_errors_path) as f:
#            motif2motif_errors = json.load(f)
#            motif2motif_errors = {tuple(key.split(' + ')): error for key, error in motif2motif_errors.items()}
#    else:
#        raise OSError("../data/motif2motif_errors.json file does not exist")
#    filenames = {m.filename for m in motifs}
#    motif2motif_errors = {(f1, f2): v for (f1, f2), v in motif2motif_errors.items() if f1 in filenames and f2 in filenames}
#    motif2motif_sims = {key: np.exp(-v) for key, v in motif2motif_errors.items()}
#    return motif2motif_sims


def remove_one(motifs, similarities: dict, cns: list):
    # High similarity means more similar
    motifs = [m for m in motifs]  # shallow copy
    assert len(motifs) >= 2
    highest = -np.inf
    highest_pair = None
    for (f1, f2), s in similarities.items():
        if s > highest:
            highest = s
            highest_pair = (f1, f2)
    m1 = [m for m in motifs if m.filename == highest_pair[0]][0]
    m2 = [m for m in motifs if m.filename == highest_pair[1]][0]
    e1 = m1.errors[np.where(cns == m1.CN)]
    e2 = m2.errors[np.where(cns == m2.CN)]
    e1.sort()
    e2.sort()
    e1 = e1[:100]
    e2 = e2[:100]
    if np.mean(e1) > np.mean(e2):
        motifs.remove(m2)
    else:
        motifs.remove(m1)
    return motifs


def run_removal(motifs, threshold, cns, norm_factors):
    CN = motifs[0].CN
    for m in motifs:
        assert m.CN == CN
    similarities = get_similarities(motifs, norm_factors, cns)
    assert len(similarities) == scipy.misc.comb(len(motifs), 2)*2
    while any(v > threshold for v in similarities.values()):
        print(f"Running removal on {len(motifs)} motifs... {max(similarities.values())}")
        motifs = remove_one(motifs, similarities=similarities, cns=cns)
        similarities = get_similarities(motifs, norm_factors, cns)
    return motifs


def separate_motifs_by_CN(motifs):
    result = defaultdict(list)
    for m in motifs:
        result[m.CN].append(m)
    return result


def main():
    with open("../data/norm_factors.json") as f:
        norm_factors = json.load(f)
    errors_path = Path("../data/motif_errors")
    choose(errors_path=errors_path, norm_factors=norm_factors)


if __name__ == "__main__":
    main()
