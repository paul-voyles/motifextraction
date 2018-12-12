import os
import json
import gzip
from typing import Union, Iterable
import numpy as np

from .aligned_data import AlignedData
from .aligned_group import AlignedGroup


def _load_json(filename, verbose=False):
    with open(filename, 'r') as f:
        all_data = json.load(f)
    if verbose:
        print(f"Loaded {filename} successfully.")
    return all_data


def _load_gzipped_json(filename, verbose=False):
    with gzip.open(filename, "rb") as f:
        d = json.loads(f.read().decode("ascii"))
    if verbose:
        print(f"Loaded {filename} successfully.")
    return d


# Use this one externally
def load_alignment_data(filename, prepend_path='', prepend_model_path='', prepend_target_path='', use_gzip=False, load_subset: Union[bool, Iterable] = False, verbose=False):
    if use_gzip or ".gz" in filename:
        all_data = _load_gzipped_json(filename, verbose=verbose)
    else:
        all_data = _load_json(filename, verbose=verbose)
    if load_subset is not False:
        all_data = [all_data[i] for i in load_subset]
    return render_alignment_data(all_data, prepend_path, prepend_model_path, prepend_target_path, verbose)


def render_alignment_data(all_data, prepend_path='', prepend_model_path='', prepend_target_path='', verbose=False):
    if prepend_path:
        prepend_model_path = prepend_path
        prepend_target_path = prepend_path
    new_data = [None for _ in range(len(all_data))]
    for i, data in enumerate(all_data):
        if not data:
            new_data[i] = None
            continue
        if isinstance(data['model'], str):
            model_file = os.path.join(prepend_model_path, data['model'])
            model_coords = None
        else:
            model_file = None
            model_coords = data['model']
        if isinstance(data['target'], str):
            target_file = os.path.join(prepend_target_path, data['target'])
            target_coords = None
        else:
            target_file = None
            target_coords = data['target']
        mapping, inverted, swapped = data.get('mapping', []), data.get('inverted', None), data.get('swapped', None)
        R = np.matrix(data.get('R', None))
        T = np.matrix(data.get('T', None))
        model_scale, target_scale = data.get('model_rescale', None), data.get('target_rescale', None)
        target_symbols, model_symbols = None, None

        aligned_fit = np.matrix(data.get('aligned_model', None))

        new_data[i] = AlignedData(
            R=R, T=T, mapping=mapping, inverted=inverted, error=data['error_lsq'], swapped=swapped,
            model_file=model_file, model_coords=model_coords, model_symbols=model_symbols, model_scale=model_scale,
            target_file=target_file, target_coords=target_coords, target_symbols=target_symbols, target_scale=target_scale,
            aligned_model_coords=aligned_fit, aligned_model_symbols=None
        )
    new_data = AlignedGroup(new_data)
    if verbose:
        print("Rendered data successfully")
    return new_data
