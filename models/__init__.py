import pandas as pd
import numpy as np

from models.constraints import MatrixConstraints
from models.dictionary_learning import DictionaryLearning
import itertools



def model_selection(P, list_K, list_lambdas, test_size, replications=100, verbose=False):
    """
    Summary of all scores
    Parameters
    ----------
    P :  ndarray
        data
    list_K :  list
        some values of K
    list_lambdas :  list
        some values of lambda
    test_size :  float
        proportion to the test size in the train/test split. In the paper we used 0.2
    verbose :  bool
        if True, print elements

    Returns
    -------
    DataFrame
        matrix of scores
    """
    n_mat = P.shape[1]
    n_test = int((test_size) * n_mat)
    Ptest = P[:, -int(n_test):]
    Ptrain = P[:, :int(n_mat - n_test)]

    df_scores = pd.DataFrame(index=list_K, columns=list_lambdas)
    elements = itertools.product(*[list_K, list_lambdas])

    for element in elements:
        if verbose:
            print(element)
        K = element[0]
        lamb = element[1]
        df_scores.loc[K, lamb] = prediction_score(Ptrain, Ptest, K, lamb, test_size, replications)

    return df_scores



def prediction_score(Ptrain, Ptest, K, lamb, test_size, replications=100):
    """
    Implementation of Algorithm 2 in the paper
    Parameters
    ----------
    Ptrain : ndarray
        data train
    Ptest : ndarray
        data test
    K : int
        number of atoms
    lamb : float
        lambda
    test_size : float
        test proportion

    Returns
    -------
    float
        metric
    """
    np.random.seed(42)
    n_test = Ptest.shape[1]
    list_errors = []

    # build train set   
    model = DictionaryLearning(K=K, r_dim=11)
    model.fit(Ptrain, lamb=lamb, max_iter=500)

    # projection of P^test on D^train
    A_test = np.linalg.inv(model.D.T @ model.D) @ model.D.T @ Ptest

    mu = model.mu.reshape(-1, 1)
    cov = np.diag(np.var(model.A, axis=1) * (1 - model.W ** 2))  # diagonal is the estimated variance of the noise
    for rep in range(replications):
        noise = np.random.multivariate_normal(np.zeros(K), cov, int(n_test) - 1).T
        A_pred = mu + (model.W.reshape(-1, 1) * A_test[:, :-1]) + noise  # without the last testing value

        P_reco = model.D@model.A
        P_pred = model.D @ A_pred

        score_train = np.linalg.norm(Ptrain - P_reco) ** 2
        score_test = np.linalg.norm(Ptest[:, 1:] - P_pred) ** 2  # without the first value
        list_errors.append(test_size*score_train + (1-test_size)*score_test)
    return np.mean(list_errors)

