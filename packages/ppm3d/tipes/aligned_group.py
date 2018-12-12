"""Datastructure for holding and working with a set of alignments."""
import numpy as np
from collections import Counter
import scipy.spatial.distance, scipy.stats
from sklearn.cluster import DBSCAN

from .cluster import Model, Positions


class AlignedGroup(object):
    def __init__(self, data):
        self.data = data
        if not isinstance(data, list) or isinstance(data, np.ndarray):
            raise Exception("The input aligned data need to be contained in an ordered list.")
        self._combined = None
        self._average = None

    def __getitem__(self, key):
        return self.data[key]

    def __len__(self):
        return self.nclusters

    @property
    def nclusters(self):
        """ The number of clusters in this Group. """
        return len(self.data)

    @property
    def successful(self):
        """ A mask to operate only on successfully aligned clusters. """
        return np.array([cluster.successful for cluster in self.data])

    @property
    def coordinates(self):
        """ A numpy array of shape (3, nclusters) with the coordination positions of each atom for each cluster. """
        return Positions( [ [a.coord[0], a.coord[1], a.coord[2]] for cluster in self.clusters for a in cluster.atoms] ).T

    def average_structure(self, force_update=False, CN=None, CN_range=None):
        """ Calculates the atom positions of the average structure of the aligned clusters in the group. """
        if not force_update and hasattr(self, '_average') and self._average is not None:
            return self._average
        else:
            self.combine(force_update=force_update, CN=CN, CN_range=CN_range)
            return self._average

    def combine(self, colorize=True, force_update=False, CN=None, CN_range=None):
        """ Combines the group of aligned clusters into a single model. """

        if not force_update and hasattr(self, '_combined') and self._combined is not None:
            return self._combined

        atom_types = {0:'Si', 1:'Na', 2:'Mg', 3:'Ti', 4:'V', 5:'Cr', 6:'Mn', 7:'Co', 8:'Fe', 9:'Ni', 10:'Cu', 11:'Zn', 12:'B', 13:'Al', 14:'Ga', 15:'C', 16:'Sn', 17:'Pb', 18:'O', -1:'Fr'}

        nsuccessful = sum(self.successful)
        coords = []
        for aligned in self.data:
            if not aligned.successful: continue
            # if not swapped (which is equivalent to swapped is False) then use rotate_target_onto_model
            # if swapped, then use rotate_model_onto_target to reproduce aligned_target
            if aligned.swapped:
                target, model = aligned.rotate_target_onto_model(rescale=False, apply_mapping=False)
            else:
                target, model = aligned.rotate_model_onto_target(rescale=False, apply_mapping=False)
            coords.extend(target.tolist())
        coords = np.array(coords)
        coords.reshape(len(coords), 3)
        assert coords.shape[1] == 3

        if CN_range is None:
            start = 6
            stop = 24
        else:
            start, stop = CN_range
        if start == stop:
            CN = start
        else:
            start, stop = start-1, stop+1  # need to go -+ one on either side for elbow method
        if CN is None:
            def opt_eps(eps):
                db = DBSCAN(eps=eps)
                if eps < 0:
                    return 1e12
                fit = db.fit(coords)
                c = Counter(fit.labels_)
                outliers = c.pop(-1, 0)
                if outliers > len(coords)*0.2 or len(c) < 8:
                    return 1e10
                return np.std(c.values())

            print("Optimizing via DBSCAN...")
            res = None
            for x0 in [0.4, 0.3, 0.7, 0.6, 0.5, 0.8, 0.9, 1.0]:
                res = scipy.optimize.minimize(opt_eps, x0, method='powell', options={'xtol': 1e-6, 'disp': True}, bounds=((0.0001, 10),))
                if res.fun < 1e9:
                    break
                else:
                    print("DBSCAN optimizer failed to converge using x0={}. Got obj value {}".format(x0, res.fun))
            EPS = res.x
            print("Chose eps={}".format(EPS))
            db = DBSCAN(eps=EPS)
            fit = db.fit(coords)
            c = Counter(fit.labels_)
            nclusters = len(c) - 1
            print("Got {} bunches and {} outliers".format(nclusters, c[-1]))
        else:
            raise NotImplemented("Specifying the CN is currently not supported.")
            nclusters = CN

        # Uncomment the below to use kmeans (it will overwrite the dbscan fit)
        #kmeans = KMeans(n_clusters=nclusters, n_init=100, max_iter=300)
        #fit = kmeans.fit(coords)

        # Also make the avg at the same time since we already have the colorized results
        c = Counter(fit.labels_)
        if len(c) < 8:
            raise RuntimeError("Too few bunches found. Terminating.")
        avg = []
        for i, z in enumerate([z for z in sorted(c) if z != -1]):
            these_coords = coords[fit.labels_ == z]
            avg.append(np.mean(these_coords, axis=0))

        avg = Positions(avg).squeeze()
        self._average = Model(symbols=['Si' for i in range(len(avg))], positions=avg)

        # Perform stdev and normalization calculations
        stdev = np.zeros((len(avg),), dtype=float)
        skew = np.zeros((len(avg),), dtype=float)
        skewtest = [None for _ in range(len(avg))]
        for i, z in enumerate([z for z in sorted(c) if z != -1]):
            dists = scipy.spatial.distance.cdist(avg[i], coords[fit.labels_ == z])[0]
            stdev[i] = np.std(dists)#/len(dists)
            skew[i] = scipy.stats.skew(dists)#/len(dists)
            skewtest[i] = scipy.stats.skewtest(dists)
            print("Bunch {} has std {} and skew {}, and the skewtest is {}".format(i, stdev[i], skew[i], skewtest[i]))
        self._combined_avg_stdev = np.mean(stdev)
        self._combined_avg_skew = np.mean(skew)
        print("Combined structure has an avg stdev of {}".format(self._combined_avg_stdev))
        print("Combined structure has an avg skew  of {}".format(self._combined_avg_skew))

        # Check to see if I got the right number of bunches/clusters
        normalized = (stdev-np.mean(stdev))/np.std(stdev)
        if np.amax(normalized) > 2:
            print("WARNING! There maybe be two bunches assigned to the same cluster.")

        if colorize:
            symbols = [atom_types[l] for l in fit.labels_]
        else:
            symbols = ['Si' for sym, coord in coords]
        coords = Positions(coords)
        print("Making combined model... does this take a long time?")
        self._combined = Model(symbols=symbols, positions=coords)
        print("Finished creating combined model.")
        return self._combined



# A function to constrain DBSCAN by nclusters.
# This allows us to get outliers, whereas kmeans can't.
#In [128]: def dbscan_with_nclusters(coords, nclusters):
#     ...:     def opt_eps_by_nclusters(eps):
#     ...:         if eps < 0:
#     ...:             return 1e12
#     ...:         db = DBSCAN(eps=eps)
#     ...:         fit = db.fit(coords)
#     ...:         c = Counter(fit.labels_)
#     ...:         n = len(c) - 1
#     ...:         return np.abs(n - nclusters)
#     ...:     for x0 in [0.4, 0.3, 0.7, 0.6, 0.5, 0.8, 0.9, 1.0]:
#     ...:         res = scipy.optimize.minimize(opt_eps_by_nclusters, x0, method='powell', options={'xtol': 1e-6, 'disp': True}, bounds=((0.0001, 10),))
#     ...:         if res.fun == 0:
#     ...:             break
#     ...:     else:
#     ...:         raise RuntimeError("dbscan could not generate a set with the correct number of clusters")
#     ...:     EPS = res.x
#     ...:     def opt_eps(eps):
#     ...:         db = DBSCAN(eps=eps)
#     ...:         if eps < 0:
#     ...:             return 1e12
#     ...:         fit = db.fit(coords)
#     ...:         c = Counter(fit.labels_)
#     ...:         outliers = c.pop(-1, 0)
#     ...:         n = len(c)
#     ...:         if outliers > len(coords)*0.2 or len(c) < 8:
#     ...:             return 1e10
#     ...:         return np.std(c.values()) + 100*np.abs(n - nclusters)
#     ...:     res = scipy.optimize.minimize(opt_eps, EPS, method='powell', options={'xtol': 1e-6, 'disp': True}, bounds=((0.0001, 10),))
#     ...:     return res