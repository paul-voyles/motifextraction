import numpy as np
import math
from itertools import combinations
from typing import Union

from .ppm3d import basic_align
from .utils import read_xyz, normalize_edge_lengths


def nCr(n, r):
    f = math.factorial
    return f(n) / f(r) / f(n-r)


def align(model: Union[str, np.ndarray],
          target: Union[str, np.ndarray],
          normalize_edges: bool = True,
          check_inverse: bool = True,
          use_combinations: Union[bool, int] = False,
          max_alignments: int = np.inf):
    """Align atom coordinates of a model to atom coordinates of a target.

    Parameters
    ----------

    model (np.array): A (N,3) shape numpy array of coordinates OR (str): filename.

    target (np.array): A (N,3) shape numpy array of coordinates OR (str): filename.

    check_inverse (bool (default True)): If True, performs a second alignment using -target. The better of the two is used.

    use_combinations (False or int (default False)): If not False, performs alignments on a subset of the targets's positions using itertools.combinations(target, len(model)+use_combinations)
                                     This helps find_map find the best mapping.
                                     Smaller values of use_combinations will create combinations so that the target and model are more similar in size.

    max_alignments (int (default np.inf)): The maximum number of combinatorical alignments that will be performed.

    """

    # If filenames were passed in, load the files
    if isinstance(target, str):
        target_fn = target
        target = read_xyz(target)
    else:
        target_fn = None
    if isinstance(model, str):
        model_fn = model
        model = read_xyz(model)
    else:
        model_fn = None

    if normalize_edges:
        target, tscale = normalize_edge_lengths(target)
        model, mscale = normalize_edge_lengths(model)
    else:
        tscale = 1.0
        mscale = 1.0

    # If len(target) > len(model), swap so that there are more points in the model than in the target.
    if target.shape[0] > model.shape[0]:
        model, target = target, model
        swapped = True
    else:
        swapped = False

    if use_combinations is not False:
        # If there are many more points in the model than in the target, the find_map function
        # has trouble finding a good mapping. We can combinatorically cut down on the number of
        # points in the model and try all combinations of them in the alignment scheme,
        # and then pick the best one.
        # The alignment seems to work well enough as long as the difference in CN is not > 3
        # Unfortunately, since this is a combinatorical problem, the number of alignments
        # increases factorially. We can cap that using the `max_alignments` parameters.
        length_of_model_combinations = min(len(target) + use_combinations, len(model))
        model_indices = np.array(range(len(model)))

        # Also don't perform more than `max_alignments` alignments
        while nCr(len(model_indices), length_of_model_combinations) > max_alignments:
            length_of_model_combinations += 1
        assert length_of_model_combinations <= len(model_indices)
        #print("nCr({}, {}) = {}".format(len(model_indices), length_of_model_combinations, nCr(len(model_indices), length_of_model_combinations)))

        best_error = np.inf
        best_results = None
        for combo in combinations(model_indices, length_of_model_combinations):
            combo = np.array(combo)
            smaller_model = model[combo, :]

            mapping, R, T, best_fit, error_dict = basic_align(smaller_model, target)
            # Unapply the itertools.combinations to the mapping
            mapping = combo[mapping]

            inverted = False
            if check_inverse:  # Also check the inverted model
                inv_mapping, inv_R, inv_T, inv_best_fit, inv_error_dict = basic_align(-smaller_model, target)
                # Unapply the itertools.combinations to the mapping
                inv_mapping = combo[inv_mapping]

                if inv_error_dict['errlsq'] < error_dict['errlsq']:
                    inverted = True
                    mapping = inv_mapping
                    R = inv_R
                    T = inv_T
                    best_fit = inv_best_fit
                    error_dict = inv_error_dict

            if error_dict['errlsq'] < best_error:
                best_error = error_dict['errlsq']
                best_results = {'mapping': mapping,
                             'R': R,
                             'T': T,
                             'best_fit': best_fit,
                             'error_dict': error_dict,
                             'inverted': inverted
                             }
        # Finished looping. Collect best results of the combinations.
        assert best_results is not None
        mapping = best_results['mapping']
        R = best_results['R']
        T = best_results['T']
        best_fit = best_results['best_fit']
        error_dict = best_results['error_dict']
        inverted = best_results['inverted']

    else:  # combinations is False
        mapping, R, T, best_fit, error_dict = basic_align(model, target)

        inverted = False
        if check_inverse:  # Also check the inverted model
            inv_mapping, inv_R, inv_T, inv_best_fit, inv_error_dict = basic_align(-model, target)
            if inv_error_dict['errlsq'] < error_dict['errlsq']:
                inverted = True
                mapping = inv_mapping
                R = inv_R
                T = inv_T
                best_fit = inv_best_fit
                error_dict = inv_error_dict
            del inv_mapping, inv_R, inv_T, inv_best_fit, inv_error_dict

    return {
            'mapping': mapping.tolist(),
            'R': R.tolist(),
            'T': T.tolist(),
            'aligned_model': best_fit.T.tolist(),
            'inverted': inverted,
            'swapped': swapped,
            'error_lsq': np.asscalar(error_dict['errlsq']),
            'error_max': np.asscalar(error_dict['errmax']),
            'target': target_fn,
            'model': model_fn,
            'model_rescale': mscale,
            'target_rescale': tscale,
           }
