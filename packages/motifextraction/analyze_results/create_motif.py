import hdbscan
import numpy as np
import pandas as pd
from ppm3d import load_alignment_data
from path import Path


def load_label(fn):
    return np.loadtxt(fn, skiprows=1).astype(int)


def cart2sph(x, y, z):
    hxy = np.hypot(x, y)
    r = np.hypot(hxy, z)
    el = np.arctan2(z, hxy)
    az = np.arctan2(y, x)
    return az, el, r


def create_motif(cluster_path: Path, label_path: Path, results_path: Path, output_data_path: Path):
    label_name = label_path.stem
    label = load_label(label_path)
    with open(label_path) as f:
        line = f.readline().strip().split()
        cn = int(line[2])
        center = int(line[3])
    aligned_data = load_alignment_data(results_path / f"{center}.xyz.json",
                                       load_subset=label,
                                       prepend_path=cluster_path
                                       )
    assert len(aligned_data) == len(label)
    nsuccessful = sum(aligned_data.successful)
    aligned_data = [a for a in aligned_data if a.successful]
    assert nsuccessful == len(aligned_data)

    aligned_clusters = []
    aligned = aligned_data[0]
    first_target = aligned.target.positions
    for aligned in aligned_data:
        target = aligned.target.positions
        model = aligned.align_model(rescale=False, apply_mapping=True)
        assert aligned.swapped is False
        assert (target == first_target).all()
        assert model.shape == target.shape
        aligned_clusters.append(np.array(model))
        assert len(target) == len(model) == cn + 1

    coords = []
    for model in aligned_clusters:
        coords.extend(model.tolist())
    coords = np.array(coords)
    coords.reshape(len(coords), 3)
    assert coords.shape[1] == 3
    atom_labels = np.zeros((coords.shape[0],))
    atom_labels.fill(np.nan)

    theta, phi, radius = cart2sph(coords.T[0], coords.T[1], coords.T[2])
    assert coords.shape[0] == theta.shape[0] == phi.shape[0] == radius.shape[0]
    print(f"Coordinate shape: {coords.shape} (~ {round(coords.shape[0] / cn)} clusters)")

    # Identify outliers -- we will ignore these atoms when calculating the centers of the bunches
    clusterer = hdbscan.HDBSCAN(min_cluster_size=15, min_samples=1).fit(coords)
    threshold = 0.99
    threshold = pd.Series(clusterer.outlier_scores_).quantile(threshold)
    outliers = np.where(clusterer.outlier_scores_ > threshold)
    atom_labels[outliers] = -1  # The outliers get labeled -1
    # Reformat the outliers so that they are structured per-model
    outliers_per_model = [[] for _ in aligned_clusters]
    count = 0
    for i, model in enumerate(aligned_clusters):
        for _ in model:
            if count in outliers[0]:
                outliers_per_model[i].append(True)
            else:
                outliers_per_model[i].append(False)
            count += 1
    outliers_per_model = np.array(outliers_per_model)
    assert (clusterer.outlier_scores_ > threshold).size == outliers_per_model.size
    assert (outliers_per_model.flatten() == (clusterer.outlier_scores_ > threshold)).all()

    # Assign labels based on the mapping.
    for i, model in enumerate(aligned_clusters):
        for j, p in enumerate(model):
            label = j
            if not outliers_per_model[i][j]:  # If this atom isn't an outlier, assign a label
                index = (i * (cn+1)) + j
                assert np.isnan(atom_labels[index])  # nan means unassigned, so here we make sure the label is unassigned
                atom_labels[index] = label
    assert (~np.isnan(atom_labels)).all()  # Make sure all atom labels have been assigned

    atom_types = {0: 'Si', 1: 'Na', 2: 'Mg', 3: 'Ti', 4: 'V', 5: 'Cr', 6: 'Mn', 7: 'Co', 8: 'Fe', 9: 'Ni', 10: 'Cu',
                  11: 'Zn', 12: 'B', 13: 'Al', 14: 'Ga', 15: 'C', 16: 'Sn', 17: 'Pb', 18: 'O', -1: 'Fr'}

    # Write the combined model to disk
    if not (output_data_path / "combined/").exists():
        (output_data_path / "combined/").makedirs()
    with open(output_data_path / f"combined/combined_{label_name}.xyz", "w") as f:
        f.write(f"{len(coords)}\n")
        f.write(f"comment\n")
        for p, label in zip(coords, atom_labels):
            f.write(f"{atom_types[label]} {p[0]} {p[1]} {p[2]}\n")

    # Calculate average structure
    centers = []
    for atom_index in range(0, (cn+1)):  # (cn+1) - 1 is the center atom
        _centers = []
        for i, (model, oliers) in enumerate(zip(aligned_clusters, outliers_per_model)):
            if not oliers[atom_index]:
                _centers.append(model[atom_index])
        _centers = np.array(_centers)
        mcenter = np.mean(_centers, axis=0)
        centers.append(mcenter)
    #print(f"Center calculated from averaging centers calculated from alignments: {mcenter}")

    # Write the averaged model to disk
    if not (output_data_path / "averaged/").exists():
        (output_data_path / "averaged").makedirs()
    with open(output_data_path / f"averaged/averaged_{label_name}.xyz", "w") as f:
        f.write(f"{len(centers)}\n")
        f.write(f"comment\n")
        for p in centers:
            f.write(f"Si {p[0]} {p[1]} {p[2]}\n")
