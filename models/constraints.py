import numpy as np


class MatrixConstraints():
    def __init__(self,  r_dim):
        """
        Constraint form
        Gx \leq h
        Qx = s
        Parameters
        ----------
        n_mat :
        r_dim :
        """
        self.r_dim = r_dim
        self.d_dim = (self.r_dim-1) * self.r_dim

        self.set_ineq = [self.positivity, self.monotonicity]  

        self.G = []
        self.h = []

        self.Q = []
        self.s = []

        return

    def positivity(self):
        """All values must be mositive"""
        G = - np.diag(np.ones(shape=(self.d_dim)))
        h = np.zeros(shape=(self.d_dim, 1))
        return G, h

    def stochastic_eq(self):
        """sum of rows equal 1"""
        Q = np.zeros(shape=(self.r_dim-1, self.d_dim))
        x = np.zeros(self.r_dim-1)
        x[0] = 1
        x = np.tile(x, self.r_dim)
        Q[0,:] = x
        for i in range(self.r_dim-2):
            Q[i+1, :] = np.roll(Q[i, :], 1)

        s = np.ones(shape=(Q.shape[0], 1))
        return Q, s

    def stochastic_ineq(self):
        """sum of rows equal 1"""
        G1, h1 = self.stochastic_eq()
        G2, h2 = self.stochastic_eq()
        G = np.concatenate([G1, -G2])
        h = np.concatenate([h1, -h2])
        return G, h


    def monotonicity(self):
        """likelihood of default"""
        G = []
        zero_block = np.zeros(shape=(self.r_dim-2, self.r_dim-1))
        one_block = np.zeros(shape=(self.r_dim-2, self.r_dim-1))
        x = np.zeros(self.r_dim-1)
        x[0] = 1
        x[1] = -1
        one_block[0,:] = x
        for i in range(1,self.r_dim-2):
            one_block[i,:] = np.roll(x, i)

        n_block_rows = (self.r_dim-1)
        n_block_ncols = self.r_dim

        for row in range(n_block_rows):
            temp_block = []
            for col_zero in range(row+1):
                temp_block.append(zero_block)
            for col_one in range(n_block_ncols-(row+1)):
                temp_block.append(one_block)
            G.append(np.concatenate(temp_block, axis=1))

        G = np.concatenate(G)
        h = np.zeros(shape=(G.shape[0], 1))
        return G, h

    def fit(self):
        # inequalities
        for func in self.set_ineq:
            mat,vec = func()
            self.G.append(mat)
            self.h.append(vec)

        self.G = np.concatenate(self.G)
        self.h = np.concatenate(self.h)

        # equalities
        self.Q, self.s = self.stochastic_eq()

        return

    def get_inequalities(self):
        return self.G, self.h

    def get_equalities(self):
        return self.Q, self.s






if __name__ == "__main__":
    NMAT = 1
    RDIM = 4
    constraints = MatrixConstraintsGT(NMAT, RDIM)
    constraints.stochastic()

