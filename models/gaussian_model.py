import numpy as np
from scipy.stats import norm
from utils.data_management import vectorized, inv_vectorized

def gaussian_transform(X, thresh=1e-16):
    """
    compute the gaussian transform on a matrix X
    Parameters
    ----------
    X : ndarray
        matrix with shape (R-1)xR. Assumes X has colsum = 1
    thresh: float
        in order to avoid numerical issues
    Returns
    -------

    """
    X_gt = np.zeros_like(X)
    # to ensure that all the rows of the first column have the same value
    X_gt[:, 0] = norm.ppf(np.ones_like(X[:, 0]) - thresh)
    for col in range(1, X.shape[1]):
        # all other columns
        # the abs(.) is to deal with 0 values
        X_gt[:, col] = norm.ppf(np.abs(X[:, col:].sum(axis=1) - thresh))
    return X_gt

class GaussianCopula():
    def __init__(self):

        self.P_Phi = None
        self.P_Phi_TTC = None
        self.P_GC = None
        self.alpha1 = 0
        self.alpha2 = 0
        self.r_dim = 0

        return

    def fit(self, P, r_dim=11):
        """

        Parameters
        ----------
        P : ndarray
            matrix of vectorized RMM in dimension d=(R-1)xR with shape (d x n_samples)
        r_dim : int
            number of columns in each RMM. Should have d=(R-1)R
        Returns
        -------

        """
        self.r_dim = r_dim
        assert (self.r_dim-1)*self.r_dim == P.shape[0], "The number of columns r_dim does not match the data shape P"

        self.fit_P_Phi(P)
        self.fit_P_Phi_TTC(P)

        n_data = self.P_Phi.shape[1]
        # least square estimate
        # Parametrisartion 1
        # ----------------
        # term1 = vectorized(self.P_Phi)
        # term2 = np.tile(self.P_Phi_TTC, n_data)
        # self.alpha2 = ((term1 * term2).sum() / (n_data * (self.P_Phi_TTC ** 2).sum()))

        # Parametrisartion 2
        # ------------------
        term1 = vectorized(self.P_Phi - self.P_Phi.mean(axis=0))
        term2 = np.tile(self.P_Phi_TTC - self.P_Phi_TTC.mean(), n_data)
        self.alpha2 = ((term1 * term2).sum() / (n_data * ((self.P_Phi_TTC-self.P_Phi_TTC.mean()) ** 2).sum()))




        # residuals
        self.alpha1 = np.array([np.sum(self.P_Phi[:, t] - self.alpha2*self.P_Phi_TTC) for t in range(n_data)]) / n_data

        # reconstruction
        self.reconstruction()

        return self

    def fit_P_Phi(self, P):
        """
        Apply Gaussian Transform to each column
        Parameters
        ----------
        P : ndarray
            matrix of vectorized RMM in dimension d=(R-1)xR with shape (d x n_samples)

        Returns
        -------

        """
        self.P_Phi = np.zeros_like(P)
        for col in range(P.shape[1]):
            # transform to matrix form each P^t
            P_t = inv_vectorized(P[:, col], nrow=self.r_dim-1, ncol=self.r_dim)
            P_Phi_t = gaussian_transform(P_t)
            self.P_Phi[:, col] = vectorized(P_Phi_t)
        return self

    def fit_P_Phi_TTC(self, P):
        """
        Apply Gaussian Transform to the P^TTC
        Parameters
        ----------
        P : ndarray
            matrix of vectorized RMM in dimension d=(R-1)xR with shape (d x n_samples)

        Returns
        -------

        """
        P_TTC = P.mean(axis=1)  # mean over the time t={1,..,T}
        self.P_Phi_TTC = vectorized(gaussian_transform(inv_vectorized(P_TTC, nrow=self.r_dim-1, ncol=self.r_dim)))
        return self

    def reconstruction(self):
        """
        computes the reconstructed RMM
        Returns
        -------

        """
        n_data = self.P_Phi.shape[1]
        term_alpha1 = np.repeat(self.alpha1.reshape(1, -1), self.P_Phi_TTC.shape[0], axis=0)
        term_alpha2 = np.repeat(self.alpha2 * self.P_Phi_TTC, n_data).reshape(self.P_Phi_TTC.shape[0], n_data)
        P_gc_phi = term_alpha1 + term_alpha2

        # define two masks
        mask1 = np.arange(self.r_dim * (self.r_dim - 1) - (self.r_dim - 1))
        mask2 = np.arange(self.r_dim - 1, self.r_dim * (self.r_dim - 1))

        P_gc = norm.cdf(P_gc_phi[mask1, :]) - norm.cdf(P_gc_phi[mask2, :])
        # add the last default column in each t=[T]
        self.P_GC = np.concatenate([P_gc, norm.cdf(P_gc_phi[-(self.r_dim - 1):, :])])
        return

    def get_coef(self):
        """
        estimated parameter
        Returns
        -------

        """
        return self.alpha2

    def get_residuals(self):
        """
        the mean residuals
        Returns
        -------

        """
        return self.alpha1


    