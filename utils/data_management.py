import numpy as np
from pathlib import Path
from models.constraints import MatrixConstraints

def load_data():
    return np.load(Path("data", "rating_migration_matrices.npy"))

def check_constraints(data):
    """compute the proportion of constraints unsatisfied in vectorized form"""
    constraints = MatrixConstraints(r_dim=11)
    constraints.fit()
    G, h = constraints.get_inequalities()
    Q, s = constraints.get_equalities()
    n_constraints = G.shape[0] + Q.shape[0]

    ineq_unsatisfied = np.minimum((np.round(G @ data, 5) > np.round(h, 5)).sum(axis=1), 1).sum()
    eq_unsatisfied = np.minimum((np.round(Q @ data, 5) != np.round(s, 5)).sum(axis=1), 1).sum()

    print("Inequality constraints unsatisfied: {} ({:.2f}%)".format(ineq_unsatisfied, ineq_unsatisfied/n_constraints * 100))
    print("Equality constraints unsatisfied: {} ({:.2f}%)".format(eq_unsatisfied, eq_unsatisfied/n_constraints*100))
    return


def vectorized(X, order='F'):
    """
    vectorize a matrix X in a given order
    Parameters
    ----------
    X : ndarray
        matrix
    order : str
        order: {‘C’, ‘F’, ‘A’, ‘K’}

    Returns
    -------

    """
    return X.flatten(order=order)

def inv_vectorized(x, nrow, ncol, order='F'):
    """
    Inverse vectorization of a vector x
    Parameters
    ----------
    x : ndarray
        vector
    nrow : int
        number of rows of the desired mat
    ncol : int
        number of col of the desired mat
    order : str
        order: {‘C’, ‘F’, ‘A’, ‘K’}

    Returns
    -------

    """
    return np.reshape(x, (nrow, ncol), order=order)
