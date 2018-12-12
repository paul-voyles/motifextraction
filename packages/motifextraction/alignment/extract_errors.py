import numpy as np
from path import Path

from ppm3d import load_alignment_data


def extract_errors(cluster_number: int, results_fn: str, save_dir: Path, cluster_dir: Path, cutoff: float, target_path: Path = None):
    if save_dir and not save_dir.exists():
        save_dir.makedirs()

    print(f"Loading {results_fn}")
    if target_path is None:
        data = load_alignment_data(results_fn, prepend_path=cluster_dir)
    else:
        data = load_alignment_data(results_fn, prepend_model_path=cluster_dir, prepend_target_path=target_path)

    errors = np.zeros((len(data), 4))
    errors.fill(np.nan)

    failed = 0
    for i, a in enumerate(data):
        if a is not None and a.successful:
            l2 = a.L2Norm()
            assert np.isclose(l2, a.error, rtol=1e-05, atol=1e-08)
            l1 = a.L1Norm()
            linf = a.LinfNorm()
            rcutoff = cutoff / (a.model_scale*0.5 + a.target_scale*0.5)
            angular = a.angular_variation(rcutoff)
        else:
            l2, l1, linf, angular = np.inf, np.inf, np.inf, np.inf
            failed += 1
        errors[i, :] = (l2, l1, linf, angular)

    print(f"Finished! {failed} alignments failed.")
    np.save(f'{save_dir}/{cluster_number}_errors.npy', errors)
