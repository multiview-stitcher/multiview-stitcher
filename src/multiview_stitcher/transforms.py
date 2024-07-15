"""
Define transforms with the same API as transforms in skimage.transform.

1) TranslationTransform
2) AffineTransform

    skimage.transform.AffineTransform (0.24.0) seems to fail to estimate in some cases.

    E.g.:

    ```
    # define some points
    pts1 = np.array([[0, 0, 0],
        [0, 0, 1],
        [0, 1, 0],
        [0, 1, 1],
        [1, 0, 0],
        [1, 0, 1],
        [1, 1, 0],
        [1, 1, 1]])

    # define matrix that inverts coordinates
    A = np.eye(4)
    A[:3, :3] = np.eye(3)[::-1]

    aff_transform = AffineTransform(matrix=A)
    pts2 = aff_transform(pts1)

    skimage_affine_transform = AffineTransform()
    skimage_affine_transform.estimate(pts1, pts2)

    print(skimage_affine_transform.params) # results in wrong params
    ```
"""

import numpy as np
from skimage.transform import (
    AffineTransform,
    ProjectiveTransform,
)

from multiview_stitcher import param_utils


class TranslationTransform(ProjectiveTransform):
    """
    Add a translation transform with the API of transforms in skimage.transform
    """

    def estimate(self, src, dst):
        translation = np.mean(dst - src, 0)
        self.params[: self.dimensionality, self.dimensionality] = translation
        return True


class AffineTransform(AffineTransform):
    """
    Add an affine estimate method to skimage.transform.AffineTransform
    """

    def estimate(self, src, dst):
        params = param_utils.affine_from_linear_affine(
            Affine_Fit(src, dst).Matrix()
        )
        self.params = params
        return True


def Affine_Fit(from_pts, to_pts):
    """
    Code from: https://elonen.iki.fi/code/misc-notes/affine-fit/

    Fit an affine transformation to given point sets.
    More precisely: solve (least squares fit) matrix 'A'and 't' from
    'p ~= A*q+t', given vectors 'p' and 'q'.
    Works with arbitrary dimensional vectors (2d, 3d, 4d...).

    Written by Jarno Elonen <elonen@iki.fi> in 2007.
    Placed in Public Domain.

    Based on paper "Fitting affine and orthogonal transformations
    between two sets of points, by Helmuth Späth (2003)."""

    q = from_pts
    p = to_pts
    if len(q) != len(p) or len(q) < 1:
        print("from_pts and to_pts must be of same size.")
        return False

    dim = len(q[0])  # num of dimensions
    if len(q) < dim:
        print("Too few points => under-determined system.")
        return False

    # Make an empty (dim) x (dim+1) matrix and fill it
    c = [[0.0 for a in range(dim)] for i in range(dim + 1)]
    for j in range(dim):
        for k in range(dim + 1):
            for i in range(len(q)):
                qt = list(q[i]) + [1]
                c[k][j] += qt[k] * p[i][j]

    # Make an empty (dim+1) x (dim+1) matrix and fill it
    Q = [[0.0 for a in range(dim)] + [0] for i in range(dim + 1)]
    for qi in q:
        qt = list(qi) + [1]
        for i in range(dim + 1):
            for j in range(dim + 1):
                Q[i][j] += qt[i] * qt[j]

    # Ultra simple linear system solver. Replace this if you need speed.
    def gauss_jordan(m, eps=1.0 / (10**10)):
        """Puts given matrix (2D array) into the Reduced Row Echelon Form.
        Returns True if successful, False if 'm' is singular.
        NOTE: make sure all the matrix items support fractions! Int matrix will NOT work!
        Written by Jarno Elonen in April 2005, released into Public Domain"""
        (h, w) = (len(m), len(m[0]))
        for y in range(h):
            maxrow = y
            for y2 in range(y + 1, h):  # Find max pivot
                if abs(m[y2][y]) > abs(m[maxrow][y]):
                    maxrow = y2
            (m[y], m[maxrow]) = (m[maxrow], m[y])
            if abs(m[y][y]) <= eps:  # Singular?
                return False
            for y2 in range(y + 1, h):  # Eliminate column y
                c = m[y2][y] / m[y][y]
                for x in range(y, w):
                    m[y2][x] -= m[y][x] * c
        for y in range(h - 1, 0 - 1, -1):  # Backsubstitute
            c = m[y][y]
            for y2 in range(y):
                for x in range(w - 1, y - 1, -1):
                    m[y2][x] -= m[y][x] * m[y2][y] / c
            m[y][y] /= c
            for x in range(h, w):  # Normalize row y
                m[y][x] /= c
        return True

    # Augement Q with c and solve Q * a' = c by Gauss-Jordan
    M = [Q[i] + c[i] for i in range(dim + 1)]
    if not gauss_jordan(M):
        # print "Error: singular matrix. Points are probably coplanar."
        return False

    # Make a result object
    class Transformation:
        """Result object that represents the transformation
        from affine fitter."""

        def Matrix(self):
            matrix = np.zeros((dim, dim))
            trans = np.zeros(dim)
            for j in range(dim):
                for i in range(dim):
                    matrix[j][i] = M[i][j + dim + 1]
                trans[j] = M[dim][j + dim + 1]
            return np.concatenate([matrix.flatten(), trans])
            # return matrix,trans

        def Transform(self, pt):
            res = [0.0 for a in range(dim)]
            for j in range(dim):
                for i in range(dim):
                    res[j] += pt[i] * M[i][j + dim + 1]
                res[j] += M[dim][j + dim + 1]
            return res

    return Transformation()
