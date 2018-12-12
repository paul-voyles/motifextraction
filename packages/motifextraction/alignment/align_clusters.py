import json
import math
from path import Path
from typing import List
import numpy as np

from ppm3d import align


def nCr(n, r):
    f = math.factorial
    return f(n) / f(r) / f(n - r)


def run_alignments(target_files: List[Path], model_files: List[Path], save_dir: Path = None, normalize_edges: bool = True, skip_if_diff_CN: bool = False):
    if save_dir and not save_dir.exists():
        save_dir.makedirs()

    for t, target_fn in enumerate(target_files):
        if save_dir:
            tail = target_files[t].name
            tail = tail + '.json'
            with open(save_dir / tail, 'w') as f:
                f.write('[')

        if skip_if_diff_CN:
            with open(target_fn) as f:
                target_CN = int(f.readline().strip())

        for m, model_fn in enumerate(model_files):
            if skip_if_diff_CN:
                with open(model_fn) as f:
                    model_CN = int(f.readline().strip())

            if skip_if_diff_CN and model_CN != target_CN:
                output = {'error_lsq': np.nan}
            else:
                output = align(model=model_fn,
                               target=target_fn,
                               normalize_edges=normalize_edges,
                               check_inverse=True,
                               use_combinations=3,
                               max_alignments=1000)

            ttail = target_files[t].name
            mtail = model_files[m].name
            output['model'] = mtail
            output['target'] = ttail
            print(mtail, ttail, output['error_lsq'])

            if save_dir:
                if m < len(model_files) - 1:
                    with open(save_dir / tail, 'a') as f:
                        f.write(json.dumps(output) + ',\n')
                else:
                    with open(save_dir / tail, 'a') as f:
                        f.write(json.dumps(output))
        if save_dir:
            with open(save_dir / tail, 'a') as f:
                f.write(']')
