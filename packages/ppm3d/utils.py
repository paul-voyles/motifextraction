import numpy as np
import scipy.spatial.distance

from .tipes import AlignedData, Cluster


def read_xyz(f):
    data = open(f).readlines()
    data.pop(0)  # Number of atoms
    data.pop(0)  # Comment
    data = np.array([[float(x) for x in line.split()[1:]] for line in data])
    return data


def normalize_edge_lengths(coordinates):
    pdists = scipy.spatial.distance.pdist(coordinates)
    mean = np.mean(pdists)
    coordinates /= mean
    return coordinates, mean


def print_outputs(outputs):
    np.set_printoptions(suppress=True)

    print("Swapped? {}\nMapping: {}\nError: {}".format(
        outputs["swapped"], outputs["mapping"], outputs["error_lsq"])
    )

    aligned = AlignedData.from_mapping(outputs)
    model = Cluster._rescale_coordinates(aligned.aligned_model, outputs["model_rescale"])
    aligned_target = Cluster._rescale_coordinates(aligned.aligned_target.positions, outputs["target_rescale"])
    atom_types = {0:'Si', 1:'Na', 2:'Mg', 3:'Ti', 4:'V', 5:'Cr', 6:'Mn', 7:'Co', 8:'Fe', 9:'Ni', 10:'Cu', 11:'Zn', 12:'B', 13:'Al', 14:'Ga', 15:'C', 16:'Sn', 17:'Pb', 18:'O', -1:'Fr'}
    print("")
    print(len(model) + len(aligned_target))
    print("Combined model and target after alignment.")
    for i, coord in enumerate(model):
        x = coord[0,0]
        y = coord[0,1]
        z = coord[0,2]
        print("{} {} {} {}".format(atom_types[i], x, y, z))
    for i, coord in enumerate(aligned_target):
        x = coord[0,0]
        y = coord[0,1]
        z = coord[0,2]
        print("{} {} {} {}".format(atom_types[i], x, y, z))
    return outputs
