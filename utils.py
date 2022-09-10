import numpy as np
import pandas as pd

sigma_cols = ['x2','xxp','xy','xyp','xp2','yxp','xpyp','y2','yyp','yp2']


def symmetrize(M):
    """Return a symmetrized version of M.

    M : Square upper or lower triangular matrix.
    """
    return M + M.T - np.diag(M.diagonal())


def mat2vec(Sigma):
    """Return vector of upper triangular elements of 4x4 matrix `Sigma`."""
    return Sigma[np.triu_indices(4)]
    
    
def vec2mat(sigma):
    """Return 4x4 symmetric matrix from 10 element vector."""
    s11, s12, s13, s14, s22, s23, s24, s33, s34, s44 = sigma
    return np.array([[s11, s12, s13, s14],
                     [s12, s22, s23, s24],
                     [s13, s23, s33, s34],
                     [s14, s24, s34, s44]])
