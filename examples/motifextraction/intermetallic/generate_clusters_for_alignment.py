""" Pseudocode:
1) this script takes in a model (specified on the command line)
2) randomly picks an atom in that model
3) creates a nearest-neighbor cluster around that atom
4) fixes any periodic boundary problems where the cluster wraps around the edge of the model
5) moves the center atom to the end of the atom list
##6) normalizes the average bond distance of the cluster to be 1.0
7) saves the cluster to a directory of your choosing
8) steps 2-6 are repeated 'num_clusters' times, and no atom can be selected twice

These clusters are used as input for Arash's rotation alignment code.
"""

import sys, os, itertools
from collections import Counter
from sortedcontainers import SortedList
from ppm3d import Model
from ppm3d import Cluster
from typing import List
from path import Path
import numpy as np
from natsort import natsorted


def generate_clusters(modelfiles):
    # Set the number of clusters to randomly select
    num_clusters = 'all'

    # Load the cutoff dictionary so that we can generate neighbors for every atom
    from motifextraction.cutoff import cutoff

    # Load the MD model
    models = []  # type: List[Model]
    for modelfile in modelfiles:
        model = Model(filename=modelfile)
        _cutoff = cutoff[tuple(Counter(model.symbols).keys())]
        models.append(model)
        print(f"Generating neighbors for model {modelfile} with cutoff {_cutoff}...")
        model.neighbors = [model.get_neighbors(model.atoms[i], cutoff=_cutoff) for i in range(model.natoms)]
    md = Model(box=models[0].box,
               symbols=list(itertools.chain.from_iterable(model.symbols for model in models)),
               positions=list(itertools.chain.from_iterable(np.array(model.positions) for model in models)),
               )

    atom_assignment = {}
    model_number = 0
    next_increment = models[model_number].natoms
    for i, atom in enumerate(itertools.chain.from_iterable(model.atoms for model in models)):
        if i == next_increment:
            model_number += 1
            next_increment += models[model_number].natoms
            print(f"Incrementing assignment to {model_number} at i = {i}")
        atom_assignment[i] = model_number
    print(Counter(atom_assignment.values()))
    print(f"Combined model has {md.natoms} atoms")

    # Directory name to save the files to
    dir = Path('data/clusters/')
    if not dir.exists():
        dir.makedirs()

    # Make a model to hold all the atoms we pull out to make sure the original model was sampled uniformly.
    # Only the center atoms are added to this model.
    if num_clusters == 'all' or num_clusters == md.natoms:
        num_clusters = md.natoms
        holding_model = md
        random_selection = False
    else:
        num_clusters = min(num_clusters, md.natoms)
        holding_model = []
        random_selection = True

    if random_selection:
        picked = SortedList(np.random.choice(range(md.atoms), size=num_clusters, replace=False))
    else:
        picked = range(md.natoms)

    print("Creating clusters...")
    cns = Counter()
    pick = 0
    for model in models:
        for i, atom in enumerate(model.atoms):
            assert model == models[atom_assignment[pick]]
            if pick not in picked:
                continue

            # Create the cluster and write to disk
            atoms = model.neighbors[i] + [atom]  # The center atom goes at the end
            # comment='cluster with center atom {0} from model {1}'.format(atom.id, model.filename),
            c = Cluster(
                symbols=[a.symbol for a in atoms],
                positions=np.array([(a.x, a.y, a.z) for a in atoms]),
                box=md.box,
            )
            c.fix_pbcs()  # This also recenters.
            if c.CN < 10:
                continue
            c.to_xyz(dir / f'{pick}.xyz')
            cns[c.CN] += 1

            pick += 1

            if random_selection:
                holding_model.append(atom)  # TODO Need to convert this to a model afterwards

    holding_model.to_xyz(filename='holding_model.xyz',
                         comment='combined model',
                         xsize=holding_model.box[0],
                         ysize=holding_model.box[1],
                         zsize=holding_model.box[2],
                         )

    print("Coordination number counts:")
    for cn, count in sorted(cns.items()):
        print(cn, count)
    with open("cns.txt", "w") as f:
        f.write("Coordination number counts:\n")
        for cn, count in sorted(cns.items()):
            f.write("{} {}\n".format(cn, count))
    return cns


def main():
    if len(sys.argv) > 1:
        modelfiles = natsorted(sys.argv[1:])
    else:
        modelfiles = natsorted(Path("modelfiles/").glob("*.xyz"))
    generate_clusters(modelfiles)


if __name__ == '__main__':
    main()
