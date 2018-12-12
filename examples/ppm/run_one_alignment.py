import sys
import numpy as np

from ppm3d import align, AlignedData


if __name__ == "__main__":
    if len(sys.argv) == 3:
        model_fn = sys.argv[1]
        target_fn = sys.argv[2]
    else:
        model_fn = 'A.xyz'
        target_fn = 'B.xyz'
    results = align(model=model_fn,
                    target=target_fn,
                    normalize_edges=True,
                    check_inverse=True,
                    use_combinations=3,
                    max_alignments=np.inf
                    )

    results = AlignedData.from_mapping(results)

    print(results.successful)
    print(results.swapped)
    print(results.inverted)
    print(results.error)
    assert np.isclose(results.error, results.L2Norm())
    print(results.L2Norm(), results.L1Norm(), results.LinfNorm(), results.angular_variation(neighbor_cutoff=3.6))
    print(results.model_scale, results.target_scale)
    print(results.mapping)
    print(results.R)
    print(results.T)
    print(results.mapping)

    print(results.aligned_model.symbols)
    print(results.aligned_model.positions)
    print(results.align_model(rescale=True, apply_mapping=True))

    print(results.align_model(rescale=False, apply_mapping=True))
