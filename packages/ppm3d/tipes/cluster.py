"""Basic datastructures for holding and working with a model or cluster of atoms."""
import os
import json
import math
import numpy as np
import scipy.spatial.distance
from collections import Counter


class Masses(object):
    def __init__(self):
        self.masses = {1:1.007947, 2:4.0026022, 3:6.9412, 4:9.0121823, 5:10.8117, 6:12.01078, 7:14.00672, 8:15.99943, 9:18.99840325, 10:20.17976, 11:22.989769282, 12:24.30506, 13:26.98153868, 14:28.08553, 15:30.9737622, 16:32.0655, 17:35.4532, 18:39.9481, 19:39.09831, 20:40.0784, 21:44.9559126, 22:47.8671, 23:50.94151, 24:51.99616, 25:54.9380455, 26:55.8452, 27:58.9331955, 28:58.69342, 29:63.5463, 30:65.4094, 31:69.7231, 32:72.641, 33:74.921602, 34:78.963, 35:79.9041, 36:83.7982, 37:85.46783, 38:87.621, 39:88.905852, 40:91.2242, 41:92.906382, 42:95.942, 43:98, 44:101.072, 45:102.905502, 46:106.421, 47:107.86822, 48:112.4118, 49:114.8183, 50:118.7107, 51:121.7601, 52:127.603, 53:126.904473, 54:131.2936, 55:132.90545192, 56:137.3277, 57:138.905477, 58:140.1161, 59:140.907652, 60:144.2423, 61:145, 62:150.362, 63:151.9641, 64:157.253, 65:158.925352, 66:162.5001, 67:164.930322, 68:167.2593, 69:168.934212, 70:173.043, 71:174.9671, 72:178.492, 73:180.947882, 74:183.841, 75:186.2071, 76:190.233, 77:192.2173, 78:195.0849, 79:196.9665694, 80:200.592, 81:204.38332, 82:207.21, 83:208.980401, 84:210, 85:210, 86:220, 87:223, 88:226, 89:227, 91:231.035882, 90:232.038062, 93:237, 92:238.028913, 95:243, 94:244, 96:247, 97:247, 98:251, 99:252, 100:257, 101:258, 102:259, 103:262, 104:261, 105:262, 106:266, 107:264, 108:277, 109:268, 110:271, 111:272, 112:285, 113:284, 114:289, 115:288, 116:292, 118:293}

    def get_mass(self, znum):
        return self.masses[znum]

    def get_znum(self, mass):
        prec1 = len(str(mass)[int(math.log10(mass))+2:])
        for z, m in self.masses.items():
            prec2 = len(str(m)[int(math.log10(m))+2:])
            p = min(prec1, prec2)
            if round(mass, p) == round(m, p): return z
        raise Exception("Mass not found!")
MASSES = Masses()


class Positions(np.matrix):
    atom_types = ['Si', 'Na', 'Mg', 'Ti', 'V', 'Be', 'Mn', 'Fe', 'P', 'Ni', 'Cu', 'S', 'B', 'He', 'Ga', 'C', 'Sn', 'Pb', 'O']

    def __init__(self, *args, **kwargs):
        super().__init__()
        self = self.view(Positions)
        assert isinstance(self, Positions)

    def apply_transformation(self, R, T, invert=False):
        if not invert:
            this = (R * self.T).T + np.tile(T, [self.shape[0], 1])
        else:
            this = (R * (self + np.tile(T, [self.shape[0], 1])).T).T
        return this.view(Positions)

    def to_xyz(self, symbols=None, comment=''):
        if symbols is None:
            symbols = ['Si' for _ in range(len(self))]
        elif isinstance(symbols, str):
                symbols = [symbols for _ in range(len(self))]
        lines = []
        natoms = len(self)
        lines.append(str(natoms))
        lines.append(comment)
        for i, row in enumerate(self):
            lines.append('{} {} {} {}'.format(symbols[i], row[0,0], row[0,1], row[0,2]))
        return '\n'.join(lines)

    def to_colorized_xyz(self, comment=''):
        lines = []
        natoms = len(self)
        lines.append(str(natoms))
        lines.append(comment)
        for i, row in enumerate(self):
            lines.append('{} {} {} {}'.format(Positions.atom_types[i], row[0,0], row[0,1], row[0,2]))
        return '\n'.join(lines)


class Atom(object):
    _symbols = {1:"H", 2:"He", 3:"Li", 4:"Be", 5:"B", 6:"C", 7:"N", 8:"O", 9:"F", 10:"Ne", 11:"Na", 12:"Mg", 13:"Al", 14:"Si", 15:"P", 16:"S", 17:"Cl", 18:"Ar", 19:"K", 20:"Ca", 21:"Sc", 22:"Ti", 23:"V", 24:"Cr", 25:"Mn", 26:"Fe", 27:"Co", 28:"Ni", 29:"Cu", 30:"Zn", 31:"Ga", 32:"Ge", 33:"As", 34:"Se", 35:"Br", 36:"Kr", 37:"Rb", 38:"Sr", 39:"Y", 40:"Zr", 41:"Nb", 42:"Mo", 43:"Tc", 44:"Ru", 45:"Rh", 46:"Pd", 47:"Ag", 48:"Cd", 49:"In", 50:"Sn", 51:"Sb", 52:"Te", 53:"I", 54:"Xe", 55:"Cs", 56:"Ba", 57:"La", 58:"Ce", 59:"Pr", 60:"Nd", 61:"Pm", 62:"Sm", 63:"Eu", 64:"Gd", 65:"Tb", 66:"Dy", 67:"Ho", 68:"Er", 69:"Tm", 70:"Yb", 71:"Lu", 72:"Hf", 73:"Ta", 74:"W", 75:"Re", 76:"Os", 77:"Ir", 78:"Pt", 79:"Au", 80:"Hg", 81:"Tl", 82:"Pb", 83:"Bi", 84:"Po", 85:"At", 86:"Rn", 87:"Fr", 88:"Ra", 89:"Ac", 90:"Th", 91:"Pa", 92:"U", 93:"Np", 94:"Pu", 95:"Am", 96:"Cm", 97:"Bk", 98:"Cf", 99:"Es", 100:"Fm", 101:"Md", 102:"No", 103:"Lr", 104:"Rf", 105:"Db", 106:"Sg", 107:"Bh", 108:"Hs", 109:"Mt", 110:"Ds", 111:"Rg", 112:"Cn", 113:"Uut", 114:"Fl", 115:"Uup", 116:"Lv", 117:"Uus", 118:"Uu"}
    numbers = {v:k for k,v in _symbols.items()}

    def __init__(self, symbol, position, id=None):
        self.id = id
        self.position = np.array(position)
        if isinstance(symbol, str):
            self._symbol = symbol
            self._number = Atom.numbers[symbol]
        elif isinstance(symbol, int):
            self._symbol = Atom._symbols[symbol]
            self._number = symbol

    @property
    def symbol(self):
        return self._symbol

    @property
    def number(self):
        return self._number

    @property
    def x(self):
        return self.position[0,0]

    @property
    def y(self):
        return self.position[0,1]

    @property
    def z(self):
        return self.position[0,2]


class Model(object):
    def __init__(self, symbols=None, positions=None, box=None, filename=None):
        self.filename = filename
        if box is None:
            self.box = np.array([None, None, None])
        else:
            self.box = np.array(box)
        if filename is None and positions is not None:
            if len(symbols) != len(positions):
                raise ValueError("symbols and positions must be the same length")
            self._positions = Positions(positions).copy()
            self.atoms = [Atom(symbol=symbol, position=position, id=i) for i, (symbol, position) in enumerate(zip(symbols, self._positions))]
        else:
            if os.path.splitext(filename)[1] == ".xyz":
                self._from_xyz(filename)
            else:
                raise ValueError("Unknown extension in filename '{}'".format(filename))

    def _from_xyz(self, filename):
        with open(filename) as f:
            natoms = int(f.readline().strip())
            self.comment = f.readline().strip()
            try:
                xs, ys, zs = self.comment.split()[:3]
                xs, ys, zs = float(xs), float(ys), float(zs)
                self.box = np.array([xs, ys, zs])
            except ValueError:
                self.box = np.array([None, None, None])
            self._positions = Positions(np.zeros((natoms, 3), dtype=float))
            self.atoms = []
            for i in range(natoms):
                symb, x, y, z = tuple(f.readline().strip().split())
                x, y, z = float(x), float(y), float(z)
                self._positions[i, 0] = x
                self._positions[i, 1] = y
                self._positions[i, 2] = z
                atom = Atom(symbol=symb, position=self._positions[i], id=i)
                self.atoms.append(atom)

    def __getitem__(self, index):
        return self.atoms[index]

    def __len__(self):
        return self.natoms

    @property
    def symbols(self):
        return np.array([atom.symbol for atom in self])

    @property
    def natoms(self):
        return len(self.atoms)

    @property
    def positions(self):
        return self._positions

    @positions.setter
    def positions(self, positions):
        self._positions = Positions(positions)
        for i in range(self.natoms):
            atom = self.atoms[i]
            atom.position = np.array(self._positions[i, :])

    def to_dict(self):
        d = {}
        if self.filename is not None:
            d['filename'] = self.filename
        d['positions'] = self._positions.tolist()
        d['symbols'] = self.symbols
        return d

    def to_json(self, **kwargs):
        return json.dumps(self.to_dict(), **kwargs)

    def to_xyz(self, filename=None, comment='', xsize=None, ysize=None, zsize=None):
        lines = []
        lines.append('{}\n'.format(self.natoms))
        comment = '{} {} {} {}'.format(xsize or '', ysize or '', zsize or '', comment).strip()
        lines.append('{}\n'.format(comment))
        for atom in self.atoms:
            lines.append('{sym} {x} {y} {z}\n'.format(sym=atom.symbol, x=atom.x, y=atom.y, z=atom.z))
        lines = ''.join(lines)
        if filename is not None:
            with open(filename, 'w') as of:
                of.write(lines)
        else:
            import sys
            sys.stdout.write(lines)
        return lines

    def recenter(self):
        xmin = np.amin(self.positions[:, 0])
        xmax = np.amax(self.positions[:, 0])
        ymin = np.amin(self.positions[:, 1])
        ymax = np.amax(self.positions[:, 1])
        zmin = np.amin(self.positions[:, 2])
        zmax = np.amax(self.positions[:, 2])
        xcenter = xmin + (xmax - xmin)/2.0
        ycenter = ymin + (ymax - ymin)/2.0
        zcenter = zmin + (zmax - zmin)/2.0
        self.positions[:, 0] -= xcenter
        self.positions[:, 1] -= ycenter
        self.positions[:, 2] -= zcenter
        for i, atom in enumerate(self.atoms):
            atom.position = self.positions[i]

    def to_dat(self, filename=None, comment='', bounds=None):
        if filename is None:
            of = sys.stdout
        else:
            of = open(filename,'w')
        of.write(comment+'\n')
        of.write('{0} atoms\n\n'.format(self.natoms))
        atomtypes = Counter(self.symbols)
        of.write('{0} atom types\n\n'.format(len(natomtypes)))
        if bounds is None:
            bounds = [
                (np.amin(self.positions[:, 0]), np.amax(self.positions[:, 0]))
                (np.amin(self.positions[:, 1]), np.amax(self.positions[:, 1]))
                (np.amin(self.positions[:, 2]), np.amax(self.positions[:, 2]))
            ]
            bounds = [(b*100, a*100) for b, a in bounds]  # TODO This manually increases the box dimensions. It won't work correctly if the cluster isn't centered.
        of.write('{0} {1} xlo xhi\n'.format(*bounds[0]))
        of.write('{0} {1} ylo yhi\n'.format(*bounds[1]))
        of.write('{0} {1} zlo zhi\n\n'.format(*bounds[2]))
        of.write('Masses\n\n')
        atomtypes = reversed(sorted(atomtypes.keys()))
        for i, z in enumerate(atomtypes):
            of.write('{0} {1}\n'.format(i+1,round(MASSES.get_mass(z),2)))
        of.write('\n')
        of.write('Atoms\n\n')
        for i, atom in enumerate(self.atoms):
            of.write('{0} {1} {2} {3} {4}\n'.format(i, atomtypes.index(atom.symbol)+1, atom.x, atom.y, atom.z))

    def get_neighbors(self, atom: Atom, cutoff: float):
        positions = np.array(self.positions)
        assert positions.shape[1] == 3
        x = atom.x - positions[:, 0]
        y = atom.y - positions[:, 1]
        z = atom.z - positions[:, 2]
        x = x - self.box[0]*np.round(x/self.box[0])
        y = y - self.box[1]*np.round(y/self.box[1])
        z = z - self.box[2]*np.round(z/self.box[2])
        dists = np.sqrt(x**2+y**2+z**2)
        return [self.atoms[i] for i in np.where((1e-6 < dists) & (dists < cutoff))[0]]


class Cluster(Model):
    def __init__(self, symbols=None, positions=None, box=None, rescaling_constant=None, filename=None):
        super(Cluster, self).__init__(symbols=symbols, positions=positions, box=box, filename=filename)
        self.rescaling_constant = rescaling_constant
        self.center_atom = self.find_center()
        # Need to reorder, putting center atom position at the end
        if self.center_atom is not None and self.center_atom is not self.atoms[-1]:
            cind = self.atoms.index(self.center_atom)
            self._positions = np.concatenate((self.positions[:cind], self.positions[cind+1:], self.positions[cind]))
            self.atoms = self.atoms[:cind] + self.atoms[cind+1:] + [self.atoms[cind]]

    def to_dict(self):
        d = super().to_dict()
        if self.rescaling_constant is not None:
            d['rescaling_constant'] = self.rescaling_constant
        return d

    @property
    def CN(self):
        has_center = self.center_atom is not None
        return self.natoms - has_center

    def find_center(self):
        """This is beautiful."""
        dist_matrix = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(self.positions))
        dist_matrix /= np.mean(dist_matrix)
        normalized = np.divide(np.std(dist_matrix, axis=0), np.std(dist_matrix))
        inverse_normalized = np.abs(1.0 - 1.0/normalized)
        for i, value in enumerate(inverse_normalized):
            # i.e. if one atom's distances to all other atoms is at least 0.5 stdevs more than all the others, return it
            if value > 0.5:
                center = self[i]
                break
        else:
            center = None
        return center

    @property
    def vp_index(self):
        if not hasattr(self, '_vp_index'):
            from .voropp import compute_index
            self._vp_index = compute_index(self.filename)
        return self._vp_index

    def dist(self, atom1, atom2, pbc=True):
        x = (atom1.x - atom2.x)
        y = (atom1.y - atom2.y)
        z = (atom1.z - atom2.z)
        if pbc:
            dx = round(x / self.box[0])
            dy = round(y / self.box[1])
            dz = round(z / self.box[2])
            x = x - self.box[0] * dx
            y = y - self.box[1] * dy
            z = z - self.box[2] * dz
            assert x <= self.box[0] / 2 + 1e-8
            assert y <= self.box[1] / 2 + 1e-8
            assert z <= self.box[2] / 2 + 1e-8
        return np.sqrt(x**2+y**2+z**2)

    def fix_pbcs(self):
        self.recenter()

        self._fix_pbcs()
        for i, atom in enumerate(self.atoms):
            self.positions[i] = atom.position
        self.recenter()

        for i, atom1 in enumerate(self.atoms):
            assert np.isclose(self.positions[i], (atom1.x, atom1.y, atom1.z)).all()
            for j, atom2 in enumerate(self.atoms):
                dpbc = self.dist(atom1, atom2)
                npbc = self.dist(atom1, atom2, pbc=False)
                if not np.isclose(dpbc, npbc):
                    self.to_xyz()
                    raise Exception("PBC fix failed!", dpbc, npbc, self.box)
        return self

    def _fix_pbcs(self, verbose=False):
        # Move all the atoms on the - side to the + side if we need to fix the cluster
        for i1, atom1 in enumerate(self.atoms):
            for i2, atom2 in enumerate(self.atoms):
                diff = np.array(atom1.position - atom2.position)[0]
                # If two atoms are exactly 0.5*box_size apart, the atom may be flipped out of the close-packed structure. We need to prevent that from happening.
                diff = diff / self.box
                for i, v in enumerate(diff):
                    if np.isclose(v, 0.5):
                        diff[i] = np.nan
                diff = np.round(diff)
                diff[np.where(np.isnan(diff))] = 0.5
                for i, v in enumerate(diff):
                    if not np.isclose(v, 0):
                        if np.isclose(v, 0.5):
                            # Decide whether or not we should move an atom
                            # TODO NO! I need to check whether the atom is closer to all other atoms or not.
                            dpbc = self.dist(atom1, atom2)
                            npbc = self.dist(atom1, atom2, pbc=False)
                            if np.isclose(dpbc, npbc):
                                continue # Don't move either atom
                        # Which atom to move??? The one that is furthest away from all the others.
                        for i3, atom3 in enumerate(self.atoms):
                            total1 = 0.
                            total2 = 0.
                            total1 += self.dist(atom1, atom3, pbc=True)
                            total2 += self.dist(atom2, atom3, pbc=True)
                        if total1 > total2:  # Then atom1 is further away
                            if verbose: print(f"Flipping {i}th position of atom {i1}")
                            atom1.position[:,i] -= self.box[i] * v
                        else:  # Then atom2 is further away
                            if verbose: print(f"Flipping {i}th position of atom {i2}")
                            atom2.position[:,i] += self.box[i] * v

                        if verbose:
                            print(diff, np.array(atom1.position - atom2.position)[0])

                x = (atom1.x - atom2.x)
                y = (atom1.y - atom2.y)
                z = (atom1.z - atom2.z)

                dx = x / self.box[0]
                dy = y / self.box[1]
                dz = z / self.box[2]
                assert dx <= 0.5 or np.isclose(dx, 0.5)
                assert dy <= 0.5 or np.isclose(dy, 0.5)
                assert dz <= 0.5 or np.isclose(dz, 0.5)

    @staticmethod
    def _rescale_coordinates(coordinates, scale):
        return coordinates * scale

    def rescale_edge_lengths(self, scale, verbose=False):
        if verbose:
            print("Rescaled by {}".format(scale))
        return Cluster._rescale_coordinates(self.positions, scale)

    def normalize_edge_lengths(self, verbose=False):
        pdists = scipy.spatial.distance.pdist(self.positions)
        mean = np.mean(pdists)
        return self.rescale_edge_lengths(1.0/mean, verbose)

    @staticmethod
    def _L2Norm(c1, c2):
        L2 = 0.0
        for i in range(len(c1)):
            L2 += ((c1[i,0] - c2[i,0])**2 +
                   (c1[i,1] - c2[i,1])**2 +
                   (c1[i,2] - c2[i,2])**2)
        return math.sqrt(L2)/len(c1)

    @staticmethod
    def _L2Norm2(c1, c2):
        assert len(c1) == len(c2)
        L2 = 0.0
        for i in range(len(c1)):
            L2 += ((c1[i,0] - c2[i,0])**2 +
                   (c1[i,1] - c2[i,1])**2 +
                   (c1[i,2] - c2[i,2])**2)
        return L2/len(c1)

    @staticmethod
    def _L1Norm(c1, c2):
        assert len(c1) == len(c2)
        L1 = 0.0
        for i in range(len(c1)):
            L1 += abs(c1[i,0] - c2[i,0]) + abs(c1[i,1] - c2[i,1]) + abs(c1[i,2] - c2[i,2])
        return L1/len(c1)

    @staticmethod
    def _LinfNorm(c1, c2):
        assert len(c1) == len(c2)
        Linf = [0.0 for i in range(len(c1))]
        for i in range(len(c1)):
            Linf[i] += (abs(c1[i,0] - c2[i,0]) +
                        abs(c1[i,1] - c2[i,1]) +
                        abs(c1[i,2] - c2[i,2]))
        return max(Linf)

    @staticmethod
    def _angle(a, b, c):
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)
        ba = a - b
        bc = c - b
        if (ba == bc).all():
            return 0.0
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        if np.abs(cosine_angle) > 1 and np.isclose(np.abs(cosine_angle), 1):  # Fix potential rounding error
                cosine_angle = 1.0 if cosine_angle > 0 else -1.0
        angle = np.arccos(cosine_angle)
        return np.degrees(angle)

    @staticmethod
    def _angular_variation(c1: "Positions", c2: "Positions", neighbor_cutoff):
        """This is a measure of the angular variation between the two clusters.
        Calculates Mean( abs(Delta(angle btwn all atoms and their neighbors, going through the center)) ).
        Both cluster must have been reordered so that self[i] corresponds directly to other[i]."""

        c1 = np.array(c1)
        c2 = np.array(c2)

        # TODO For now, assume the center atom is the last one
        center1_index = len(c1) - 1
        center2_index = len(c2) - 1
        center1 = c1[center1_index]
        center2 = c2[center2_index]
        assert center1_index == center2_index

        # Generate neighbors using pdist and the array for both c1 and c2
        c1_dist_matrix = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(c1))
        c2_dist_matrix = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(c2))
        dist_matrix = (c1_dist_matrix + c2_dist_matrix) / 2.0  # Calculate the average atom-to-atom distances
        neighbors = []
        for index_pair, x in np.ndenumerate(dist_matrix):
            if x < neighbor_cutoff and center1_index not in index_pair:
                neighbors.append(index_pair)

        # Use those neighbors to calculate angles for c1 and c2 simultaneously, and calculate abs(Delta(angles))
        # Also calculate and return the mean
        mean = 0.
        for neighbor_pair in neighbors:
            i, j = neighbor_pair
            xi1, yi1, zi1 = c1[i,0], c1[i,1], c1[i,2]
            xi2, yi2, zi2 = c2[i,0], c2[i,1], c2[i,2]
            xj1, yj1, zj1 = c1[j,0], c1[j,1], c1[j,2]
            xj2, yj2, zj2 = c2[j,0], c2[j,1], c2[j,2]
            a1 = Cluster._angle((xi1, yi1, zi1), center1, (xj1, yj1, zj1))
            a2 = Cluster._angle((xi2, yi2, zi2), center2, (xj2, yj2, zj2))
            mean += abs(a1 - a2)
        if len(neighbors) == 0:
            return np.inf
        mean /= len(neighbors)
        return mean

