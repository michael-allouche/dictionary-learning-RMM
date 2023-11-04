import numpy as np
import pandas as pd
from scipy.stats import norm
from IPython.display import display, Latex

"""
Generate matrices following the copula model:
X^t = \rhoZ^t + \sqrt{1 - \rho^2}\eps^t
with Z^t = \kappaZ^t-1 + \eps^z_t

Generate with probit model without row normalization

vectorization by columns
"""

class MatrixGenerator():
    def __init__(self, n_mat, r_dim, Z_t=0, rho=0.5):
        np.random.seed(123)

        self.n_mat = n_mat  # number of matrices to generate: value of T
        self.r_dim = r_dim  # R dimension of a matrix with shape [R-1]x[R]
        self.d_dim = (self.r_dim - 1) * self.r_dim
        self.rho = rho  # correlation factor between the economic and the financial sector
        self.Z_t = Z_t  # initialize first value of Z_t
        self.list_mat_ttc = []  # save all matrices in a list: mat {P}_{t=0}^T
        self.list_eco_series = []  # save all Z^t

        self.kappa = np.exp(-np.log(2) / 20)  # speed mean reversion assuming economic cycle of 5 years and matrices issued quarterly
        self.save_eco_series()
        self.mat_ttc = self.initialize_mat_ttc()
        self.mat_ttc_gt = self.gaussian_transform(self.mat_ttc)
        self.full_mat_pit = None  # stacked pit matrices by columns in shape ((r-1*r), T)
        self.full_mat_pit_gt = None  # gaussian transform version
        return

    def initialize_mat_ttc(self, thresh=1e-10):
        """:return P^{TTC} given the following model (before normalization):
            2N,                 [(N-1)/2]*[1/2],   (N-2)/3,    (N-3)/4
            [(N-1)/2]*[2/1],    2(N-1),            (N-2)/2,    (N-3)/3
            (N-2)/3,            (N-2)/2,           2(N-2),     (N-3)/2
        ******************************************************
            2(N-i),             [(N-j)/(1+j)] * [i/j],  ...,
            [(N-i)/(1+i)] * [j/i],     ...,             ...,
            ...,                       ...,             ...
        """
        pttc = np.zeros((self.r_dim, self.r_dim))
        for col in range(1, self.r_dim):  # upper diagonale
            for row in range(col):
                pttc[row][col] = (self.r_dim - col) / (col + 1 - row) * ((row + 1) / col)

        for row in range(1, self.r_dim):  # lower diagonale
            for col in range(row):
                pttc[row][col] = (self.r_dim - row) / (row + 1 - col) * ((col + 1) / (row))

        np.fill_diagonal(pttc, [(2 * row) for row in range(self.r_dim)][::-1])

        pttc = pttc / pttc.sum(axis=1).reshape(-1, 1)  # normalized
        pttc = pttc[:-1, :]  # remove last row

        # avoid numerical issues in 0 and 1
        pttc = np.abs(pttc - thresh)

        return pttc

    def update_economic_serie(self):
        """
        Z_t = kappa * Z_t-1 + epsilon
        :return: update the serie of the economic situation
        """
        # noise = np.random.normal(0, 1)  # idiosyncratic risk: proba of being up/down graded
        noise = np.random.normal(0, np.sqrt(1-self.kappa**2))  # idiosyncratic risk: proba of being up/down graded
        self.Z_t = (self.kappa * self.Z_t) + noise
        self.save_eco_series()  # save Z^t
        return

    def save_eco_series(self):
        """save all Z^t"""
        self.list_eco_series.append(self.Z_t)
        return

    def fit(self):
        """
        :return: train matrix {P}_{t=0}^T where all P^t are in vectorize form
        """
        # np.random.seed(123)
        list_mat_pit = []
        list_mat_pit_gt = []
        for idx_mat in range(self.n_mat):
            self.update_economic_serie()  # update Z_t
            mat_pit = np.zeros(shape=(self.r_dim - 1, self.r_dim))  # init a new matrix P^{PIT}
            # mat_pit_gt = np.zeros_like(mat_pit)  # gaussian transform
            for row in range(self.r_dim - 1):
                for col in range(self.r_dim):
                    p_ttc_j = self.mat_ttc[row, col:].sum()
                    p_ttc_j_prime = self.mat_ttc[row, col + 1:].sum()

                    mat_pit[row, col] = (self.get_proba_default(p_ttc_j) - self.get_proba_default(p_ttc_j_prime))
                    # adding noise
                    # -----------
                    mat_pit[row, col] *= (1+np.random.uniform(-0.01, 0.01))
            # project to stochastic matrices
            # -----------------------------
            mat_pit = mat_pit/np.sum(mat_pit, axis=1).reshape(-1, 1)


            list_mat_pit.append(mat_pit.flatten(order="F"))  # vectorize row major
            list_mat_pit_gt.append(self.gaussian_transform(mat_pit).flatten(order="F"))

        self.full_mat_pit = np.array(list_mat_pit).T  # entire matrix PIT in vectorize form (by rows)
        self.full_mat_pit_gt = np.array(list_mat_pit_gt).T
        return

    def get_proba_default(self, p_ttc):
        """
        :param p_ttc: probability of default ttc
        :return: given the model P^t_{ij} = prob_def_j - prob_def_j_prime
        """
        num = (self.rho * self.Z_t) + norm.ppf(np.minimum(p_ttc, 1.0))  # \rho * Z^t + cdf^-1(p_ttc) and the minimum() ensure that p_ttc <=1
        denum = np.sqrt((1 - np.square(self.rho)))  # (1-\rho^2)^(1/2)
        return norm.cdf(num/denum)


    def show_pit(self):
        """
        n_mat_to_show: number of matrix to show
        :return: all P^t in matrix form
        """
        for idx_mat in range(self.n_mat):
            display(Latex("$P^{}: \quad Z^{}={}$".format(idx_mat, idx_mat, np.round(self.list_eco_series[idx_mat],3))))
            display(self.vec_inverse(self.full_mat_pit[:, idx_mat]))
        return

    def show_mat_pit(self):
        display(Latex("$\mathbf P$"))
        display(pd.DataFrame(self.full_mat_pit))
        return

    def show_ttc(self):
        display(Latex("$P^{TTC}$:"))
        df = pd.DataFrame(np.round(self.mat_ttc, 3))
        display(df)
        return

    def get_full_mat_pit(self):
        return self.full_mat_pit

    def get_rdm_pit(self):
        """build a random pit matrix with just the stochastic constraints"""
        np.random.seed(40)
        out = np.zeros(shape=(self.d_dim, self.n_mat))
        for mat in range(self.n_mat):
            # generate random matrix in U(0,1)
            rdm_mat = np.random.uniform(size=(self.r_dim-1, self.r_dim))
            # normalize by rows (stochastic matrix)
            rdm_mat /= np.sum(rdm_mat, axis=1).reshape(-1, 1)
            # append in vectorized form
            out[:, mat] = rdm_mat.flatten()
        return out


    def vec_inverse(self, x):
        """
        rows major order
        Parameters
        ----------
        x : ndarray

        Returns
        -------

        """
        matrix_form = x.reshape(self.r_dim - 1, self.r_dim, order="F")
        print("Sum of rows: ",matrix_form.sum(axis=1))
        return pd.DataFrame(matrix_form)


    def gaussian_transform_matrix(self, X, thresh=1e-16):
        """
        compute the gaussian transform on a matrix of vectorized matrices in row order
        Parameters
        ----------
        X : ndarray
        thresh: float
            avoid numerical issues
        Returns
        -------

        """
        X_gt_list = []
        # n_mat = X.shape[1]
        for t in range(self.n_mat):
            xt_vec = X[:, t]
            xt = np.reshape(xt_vec, (self.r_dim - 1, self.r_dim), order="F")
            X_gt_list.append(self.gaussian_transform(xt, thresh).flatten(order='F'))
        X_gt = np.array(X_gt_list).T
        return X_gt

    @staticmethod
    def gaussian_transform(X, thresh=1e-16):
        """
        compute the gaussian transform on a matrix X
        Parameters
        ----------
        X : ndarray
        thresh: float
            avoid numerical issues
        Returns
        -------

        """

        X_gt = np.zeros_like(X)
        # X_gt[:, 0] = norm.ppf(np.ones_like(X_gt[:, 0]) - thresh)  # assume X has col sum = 0 or 1
        X_gt[:, 0] = norm.ppf(X.sum(axis=1))  # assume X has no col sum = 0 or 1
        for col in range(1, X.shape[1]):
            # X_gt[:, col] = norm.ppf(np.abs(X[:, col:].sum(axis=1) - thresh))  # assume X has col sum = 0 or 1
            X_gt[:, col] = norm.ppf(X[:, col:].sum(axis=1))  # assume X has no col sum = 0 or 1
        return X_gt



if __name__ == '__main__':
    N_DATA = 100  # number of matrices : value of T
    R_DIM = 11  # Dim of the matrix [R-1]x[R]

    matrix_generator = MatrixGenerator(n_mat=N_DATA, r_dim=R_DIM)
    matrix_generator.fit()
    # data = matrix_generator.full_mat_pit
    # matrix_generator.show_pit()

    # matrix_generator.get_rdm_pit()