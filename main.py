from models import DictionaryLearning
from utils.data_management import load_data

if __name__ == '__main__':

    P = load_data()
    K = 2
    N_MAT = P.shape[1]
    R_DIM = 11
    LAMB = 5
    MAX_ITER = 500

    n_test = int(0.2 * N_MAT)
    n_train = int(N_MAT - n_test)

    Ptrain = P[:, :n_train]
    Ptest = P[:, -int(n_test):]


    model = DictionaryLearning(K=K, r_dim=R_DIM)
    model.fit(Ptrain, lamb=LAMB,  max_iter=MAX_ITER)

    print('Optimization finished !')
