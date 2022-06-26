import pandas as pd
import numpy as np

from models.constraints import MatrixConstraints
from models.dictionary_learning import DictionaryLearning
import itertools



def model_selection(P, list_K, list_lambdas, test_size):
    n_mat = P.shape[1]
    n_test = int((test_size) * n_mat)
    Ptest = P[:, -int(n_test):]
    Ptrain = P[:, :int(n_mat - n_test)]

    df_scores = pd.DataFrame(index=list_K, columns=list_lambdas)
    elements = itertools.product(*[list_K, list_lambdas])

    for element in elements:
        print(element)
        K = element[0]
        lamb = element[1]
        df_scores.loc[K, lamb] = prediction_score(Ptrain, Ptest, K, lamb)

    return df_scores



def prediction_score(Ptrain, Ptest, K, lamb):
    np.random.seed(42)
    n_test = Ptest.shape[1]

    # build train set   
    model_train = DictionaryLearning(K=K, r_dim=11)
    model_train.fit(Ptrain, lamb=lamb, max_iter=500)

    # projection of P^test on D^train
    A_test = np.linalg.inv(model_train.D.T @ model_train.D) @ model_train.D.T @ Ptest

    mu = model_train.mu.reshape(-1, 1)
    cov = np.diag(np.var(model_train.A, axis=1) * (1 - model_train.W ** 2))  # diagonal is the estimated variance of the noise
    noise = np.random.multivariate_normal(np.zeros(K), cov, int(n_test) - 1).T
    A_pred = mu + (model_train.W.reshape(-1, 1) * A_test[:, :-1]) + noise  # without the last testing value

    P_pred = model_train.D @ A_pred
    score = np.linalg.norm(Ptest[:, 1:] - P_pred) ** 2

    return score

