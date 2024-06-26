import numpy as np
from models.constraints import MatrixConstraints
from cvxopt import matrix
from cvxopt import solvers
import seaborn as sns
import random

# optimization with verbose=False
solvers.options['show_progress'] = False
sns.set_style("whitegrid", {'grid.linestyle': '--'})


class DictionaryLearning():
    def __init__(self, K, r_dim, verbose=False):
        self.K = K  # truncature
        self.r_dim = r_dim  # nb of columns in the RMM
        self.d_dim = (self.r_dim-1) * self.r_dim  # dimension of the problem
        self.n_mat = None  # number of matrices (T)
        self.verbose = verbose  # printing


        # initialization of ojects to optimize
        self.D = np.zeros(shape=(self.d_dim, self.K))  # dictionary
        self.A = None  # codings
        self.W = np.ones(shape=self.K)  # AR parameter
        self.mu = None  # constant AR

        # fit constraints
        self.constraints = MatrixConstraints(self.r_dim)
        self.constraints.fit()

        if self.verbose:
            print("Constraints encoded !")
            print("-"*40)
        # positivity constraints for alpha
        # --------------------------------------
        self.G_pos = None
        self.h_pos = None
        # --------------------------------------
        # dictionary constraints
        self.G, self.h = self.constraints.get_inequalities()
        self.Q, self.s = self.constraints.get_equalities()
        if self.verbose:
            print("dim(G): ({}, {}), dim(h): ({}, {})".format(self.G.shape[0], self.G.shape[1], self.h.shape[0],
                                                              self.h.shape[1]))
            print("dim(Q): ({}, {}), dim(s): ({}, {})".format(self.Q.shape[0], self.Q.shape[1], self.s.shape[0],
                                                              self.s.shape[1]))

            print("-" * 40)


        # init storing list
        self.list_total_objective = []
        self.list_reconstruction = []
        self.list_regularization = []
        return

    def fit(self, P, lamb=0., max_iter=100):
        """

        Parameters
        ----------
        P : ndarray
            data matrix with shape (dimension x n_samples)
        lamb : float
            regularization parameter
        max_iter : int
            maximum outer iterations for dictionary learning

        Returns
        -------

        """
        np.random.seed(42)
        random.seed(42)
        self.n_mat = P.shape[1]
        self.A = np.zeros(shape=(self.K, self.n_mat))  # codings
        # positivity constraints for alpha
        # --------------------------------------
        self.G_pos = -np.identity(self.n_mat)
        self.h_pos = np.zeros(shape=self.n_mat)
        # --------------------------------------
        iteration = 0

        # initialize the dictionary with random signals from the data
        self.D = P[:, random.sample(range(self.n_mat), self.K)]

        while iteration < max_iter:
            # update codings
            self.fit_codings(P=P, lamb=lamb)

            # update dictionary
            self.fit_dictionary(P)

            # update W
            self.fit_W()

            # compute constant mu
            self.mu = (1 - self.W) * np.mean(self.A, axis=1)

            reconstruction, regularizaton, objective = self.get_rro(P, lamb)

            self.list_total_objective.append(objective)
            self.list_reconstruction.append(reconstruction)
            self.list_regularization.append(regularizaton)

            iteration += 1

        return self

    ##################################################################
    #                       Dictionary update
    ##################################################################

    def fit_dictionary(self, P):
        """
        Fit dictionary
        Parameters
        ----------
        P : ndarray
            data matrix
        method : str
            algo for dictionary update
        Returns
        -------

        """
        for k in range(1, self.K + 1):
            P_k = P - (np.delete(self.D, k - 1, 1) @ np.delete(self.A, k - 1, 0))  # update \tilde P(k)
            p_k = self.vectorized(P_k).reshape(-1, 1)
            A_k = self.circulent_A(k)
            H_k = 2 * (A_k.T @ A_k)
            c_k = -2 * (A_k.T @ p_k)

            sol = solvers.qp(matrix(H_k, tc='d'), matrix(c_k, tc='d'),
                          matrix(self.G, tc='d'), matrix(self.h, tc='d'),
                          matrix(self.Q, tc='d'), matrix(self.s, tc='d'))
            self.D[:, k - 1] = np.array(sol["x"]).ravel()
        return



    ##################################################################
    #                           Codings update
    ##################################################################
    def fit_codings(self, P, lamb=0.):
        """fit codings"""
        for k in range(1, self.K + 1):
            P_k = P - (np.delete(self.D, k - 1, 1) @ np.delete(self.A, k - 1, 0))  # update \tilde P(k)
            p_k = self.vectorized(P_k).reshape(-1, 1)
            D_k = self.circulent_D(k)
            H_k = 2 * (D_k.T @ D_k)
            c_k = - 2 * (D_k.T @ p_k)
            R = self.regularization(k)
            R = (R + R.T)  # symmetrization for the solver
            H = H_k + lamb*R

            sol = solvers.qp(matrix(H, tc='d'), matrix(c_k, tc='d'), matrix(self.G_pos, tc='d'), matrix(self.h_pos, tc='d'))
            self.A[k - 1, :] = np.array(sol["x"]).ravel()
        return

    ##################################################################
    #                           W update
    ##################################################################
    def fit_W(self):
        """with mean constraint in the regularization"""
        for k in range(1, self.K + 1):
            time_series = [self.A[k - 1, t] for t in range(self.n_mat)]
            mean = np.mean(time_series)
            time_series = [time_series[t] - mean for t in range(self.n_mat)]
            nominator = sum([time_series[t] * time_series[t - 1] for t in range(1, self.n_mat)])
            # denominator = sum([time_series[t] ** 2 for t in range(0, self.n_mat - 1)])
            norm1 = sum([time_series[t]**2 for t in range(1, self.n_mat)])
            norm2 = sum([time_series[t-1]**2 for t in range(1, self.n_mat)])
            denominator = np.sqrt(norm1*norm2)  # for numerical stability
            self.W[k - 1] = nominator / (denominator)
        return


    def regularization(self, k):
        """get rehularization term \lamb * <A_k, H A_k>"""
        mat = np.zeros(shape=(self.n_mat, self.n_mat))
        for i in range(1, self.n_mat - 1):
            mat[i, i] = (1 + self.W[k - 1] ** 2)  # diagonal
            mat[i, i - 1] = -2 * self.W[k - 1]  # lower diagonal
        mat[0, 0] = (self.W[k - 1] ** 2)
        mat[self.n_mat - 1, self.n_mat - 1] = 1
        mat[self.n_mat - 1, self.n_mat - 2] = -2 * self.W[k - 1]
        auxiliar = (-1 / self.n_mat) * np.ones(shape=(self.n_mat, self.n_mat))
        for i in range(self.n_mat):
            auxiliar[i, i] = 1 - 1 / self.n_mat
        H = auxiliar @ (mat @ auxiliar)
        return H

    # def predicted_codings(self, X):
    #     """
    #     Coding predictions \mu_k + X*w_k
    #     """
    #     mean_est = self.A.mean(axis=1).reshape(-1, 1) * (1 - self.W.reshape(-1, 1))
    #     return mean_est + (self.W.reshape(-1, 1) * X)

    def predicted_codings(self, X, steps, seed):
        """
        Coding predictions \mu_k + X*w_k + epsilon_k
        """
        np.random.seed(seed)
        mean_est = self.A.mean(axis=1).reshape(-1, 1) * (1 - self.W.reshape(-1, 1))
        cov = np.diag(np.var(self.A, axis=1) * (1 - self.W ** 2))  # diagonal is the estimated variance of the noise

        for step in range(steps):
            eps = np.random.multivariate_normal(np.zeros(self.K), cov, 1).T
            X = mean_est + (self.W.reshape(-1, 1) * X) + eps
        return X


# -------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------

    ##################################################################
    #           Auxiliary functions
    ##################################################################

    def get_rro(self, P, lamb):
        """
        get the reconstruction, regularization and objective of the AR1-L2 model
        Returns
        -------

        """

        reconstruction = np.linalg.norm(P - self.D @ self.A) ** 2
        mean_est = self.A.mean(axis=1).reshape(-1,1)
        regularization = np.sum((self.A[:, 1:] - mean_est - (self.W.reshape(-1, 1) * (self.A[:, :-1] - mean_est))) ** 2)
        objective = reconstruction + lamb * regularization
        return reconstruction, regularization, objective


    def circulent_A(self, k):
        """
        Compute \tilde A_k int the paper
        Parameters
        ----------
        k : int
            k-th atom

        Returns
        -------

        """
        mat = np.zeros(shape=(self.d_dim*self.n_mat, self.d_dim))
        indices = np.arange(0, self.d_dim*self.n_mat -2, self.d_dim)
        for time, i in enumerate(indices):
            mat[i:i+self.d_dim, :] = np.diag(np.ones(self.d_dim) * self.A[k-1, time])
        return mat

    def circulent_D(self, k):
        """
        Compute \tilde D_k matrix in the paper
        Parameters
        ----------
        k : int
            k-th atom
        Returns
        -------

        """
        mat = np.zeros(shape=(self.d_dim*self.n_mat, self.n_mat))
        indices = np.arange(0, self.d_dim*self.n_mat - 2, self.d_dim)
        for time, i in enumerate(indices):
            mat[i:i+self.d_dim, int(i/self.d_dim)] = self.D[:, k-1]
        return mat


    @staticmethod
    def vectorized(x, order='F'):
        """vectorization with column major"""
        return x.flatten(order=order)
    @staticmethod
    def inv_vectorized(x, nrow, ncol, order='F'):
        """inverse vectorization with column major"""
        return np.reshape(x, (nrow, ncol), order=order)



