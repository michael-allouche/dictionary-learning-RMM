import numpy as np

from models import model_selection, DictionaryLearning
from utils.data_management import load_data

# TODO: argparse

if __name__ == '__main__':

    P = load_data()
    K = 2
    N_MAT = P.shape[1]
    R_DIM = 11
    LAMB = 5
    # MAX_ITER = 500

    n_test = int(0.2 * N_MAT)
    n_train = int(N_MAT - n_test)

    Ptrain = P[:, :n_train]
    Ptest = P[:, -int(n_test):]


    #df_selection = model_selection(P, list_K=[11, 12, 13, 14, 15],
                                   # list_lambdas=[0.1, 0.25, 0.5, 0.75, 0.9, 1, 5, 10],
                                   # test_size=0.2)

    # df_selection.to_csv("ckpt/model_selection_11-14.csv")

    model = DictionaryLearning(K=K, r_dim=R_DIM)
    model.fit(Ptrain, lamb=LAMB,  max_iter=500)

    print('Optimization finished !')
