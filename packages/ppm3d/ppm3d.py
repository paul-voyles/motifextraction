"""The raw ppm3d code written by Srikanth Patala and Arash Banadaki. (+ a few minor setup changes by Jason Maldonis)"""
import os
import numpy as np

import ctypes
import numpy.ctypeslib as npct

module_directory = os.path.dirname(os.path.realpath(__file__))
module_directory = os.path.join(module_directory, '_ppm3d')
cwd = os.getcwd()
os.chdir(module_directory)

libm = np.ctypeslib.load_library('point_match.so', module_directory)
# input type ppm3d function
# must be a double array, with single dimension that is contiguous
array_1d_integer = npct.ndpointer(dtype=ctypes.c_int, ndim=1, flags='CONTIGUOUS')
array_2d_double = npct.ndpointer(dtype=np.double, ndim=2, flags='CONTIGUOUS')
# setup the return types and argument types
libm.point_match.argtypes = [array_2d_double, ctypes.c_int, array_2d_double, ctypes.c_int, ctypes.c_int, array_1d_integer, ctypes.c_bool]
libm.point_match.restype = None

os.chdir(cwd)


def find_map(model, target, k=3, n_choosek_flag=True):
    model = np.array(model, np.float)
    target = np.array(target, np.float)
    output_map = np.zeros((model.shape[0],), dtype=ctypes.c_int)

    libm.point_match(model, model.shape[0], target, target.shape[0], k, output_map, n_choosek_flag)
    return output_map - 1


def absor(A, B, do_scale=False, weight=None):
    if weight is None:
        weight = []
    A = np.array(A.T)
    B = np.array(B.T)
    in_dim = len(A)
    if in_dim != len(B):
        print('The number of points to be registered must be the same')
        raise Exception
    if not weight:
        sumwts = 1

        lc = np.mean(A, axis=1)
        rc = np.mean(B, axis=1)  # Centroids
        left = A - lc.reshape(in_dim, 1)  # Center coordinates at centroids
        right = B - rc.reshape(in_dim, 1)
    else:
        weights = np.array(weight, dtype=float).reshape(in_dim, 1)
        sumwts = np.sum(weights)
        weights = abs(weights) / sumwts
        sqrtwts = np.sqrt(weights).reshape(1, in_dim)

        lc = np.dot(A, weights)
        rc = np.dot(B, weights)  # Weighted centroids

        left = A - lc
        left = left * sqrtwts
        right = B - rc
        right = right * sqrtwts

    M = np.dot(left, right.T)
    # Compute rotation matrix
    if in_dim == 2:
        Nxx = M[0, 0] + M[1, 1]
        Nyx = M[0, 1] - M[1, 0]

        N = np.array([[Nxx, Nyx], [Nyx, -Nxx]])
        D, V = np.linalg.eig(N)
        emax = np.argmax(np.real(V))
        q = np.real(V[:, emax])  # %Gets eigenvector corresponding to maximum eigenvalue
        # q = real(q)  #%Get rid of imaginary part caused by numerical error

        if q[1] >= 0:
            q1 = 1
        q = q * np.sign(q[1] + q1)  # Sign ambiguity
        q /= np.linalg.norm(q)

        R11 = q[0] ** 2 - q[1] ** 2
        R21 = np.prod(q) * 2
        R = np.array([[R11, -R21], [R21, R11]])  # %map to orthogonal matrix
    elif in_dim == 3:
        Sxx = M[0, 0]
        Syx = M[1, 0]
        Szx = M[2, 0]
        Sxy = M[0, 1]
        Syy = M[1, 1]
        Szy = M[2, 1]
        Sxz = M[0, 2]
        Syz = M[1, 2]
        Szz = M[2, 2]

        N = np.array([[(Sxx + Syy + Szz), (Syz - Szy), (Szx - Sxz), (Sxy - Syx)],
                      [(Syz - Szy), (Sxx - Syy - Szz), (Sxy + Syx), (Szx + Sxz)],
                      [(Szx - Sxz), (Sxy + Syx), (-Sxx + Syy - Szz), (Syz + Szy)],
                      [(Sxy - Syx), (Szx + Sxz), (Syz + Szy), (-Sxx - Syy + Szz)]])

        D, V = np.linalg.eig(N)
        emax = np.argmax(np.real(D))
        q = np.real(V[:, emax])
        ii = np.argmax(abs(q))
        sgn = np.sign(q[ii])
        q = q * sgn  # Sign ambiguity

        quat = q[:]
        nrm = np.linalg.norm(quat)
        if not nrm:
            raise ValueError('Quaternion distribution is 0')

        quat /= nrm

        q0 = quat[0]
        qx = quat[1]
        qy = quat[2]
        qz = quat[3]
        v = quat[1:4]

        Z = np.array([[q0, -qz, qy],
                      [qz, q0, -qx],
                      [-qy, qx, q0]])

        R = v.reshape(in_dim, 1) * v + np.dot(Z, Z)

    else:
        raise ValueError('Points must be either 2D or 3D')

    if do_scale:
        sss = np.sum(right * np.dot(R, left)) / np.sum(np.square(left))
        t = rc - np.dot(R, (lc * sss).T).T
    else:
        sss = 1
        t = rc - np.dot(R, lc.reshape(in_dim, 1)).T[0]
    if in_dim == 2:
        regParams = {'R': R, 't': t, 's': sss,
                     'M': np.vstack(([np.hstack((sss * R, t.reshape(in_dim, 1))), [0, 0, 1]])),
                     'theta': np.arctan2(q[1], q[0]) * 360 / np.pi}
    else:
        regParams = {'R': R, 't': t, 's': sss,
                     'M': np.vstack(([np.hstack((sss * R, t.reshape(in_dim, 1))), [0, 0, 0, 1]])),
                     'q': q / np.linalg.norm(q)}
    Bfit = np.dot((sss * R), A) + t.reshape(in_dim, 1)
    err = np.sqrt(np.sum(np.square(Bfit - B), axis=0))
    if weight:
        err = err * sqrtwts
    ErrorStats = {'errlsq': (np.linalg.norm(err) * np.sqrt(sumwts))/A.shape[1], 'errmax': max(err)}
    return regParams, Bfit, ErrorStats


def basic_align(model, target):
    # Aligns the model[mapping] onto the target.
    #  i.e.  R(M[mapping]) =~= T.
    # Since the model can be larger than the target, the mapping MUST get applied to the model in order to reduce the
    #  number of points in it.
    # The (potentially smaller) remapped model then gets aligned onto the target.
    assert len(model) >= len(target)
    mapping = find_map(target, model)  # will gave len(model)
    remapped_model = model[mapping]
    assert len(remapped_model) == len(target)
    data, best_fit, errors = absor(remapped_model, target)
    best_fit = np.array(best_fit).T
    R = data['R']
    T = data['t']
    # Now best_fit ~= target[mapping]
    return mapping, R, T, best_fit, errors
