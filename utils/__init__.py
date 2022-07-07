from models import DictionaryLearning
import numpy as np
from statsmodels.tsa.ar_model import AutoReg


def regularization_benefit(P_train, P_test, K, lamb, r_dim=11):
    # regularized model
    model_reg = DictionaryLearning(K=K, r_dim=r_dim)
    model_reg.fit(P_train, lamb=lamb, max_iter=500)

    # non-regularized model
    model_pca = DictionaryLearning(K=K, r_dim=r_dim)
    model_pca.fit(P_train, lamb=0., max_iter=500)

    # projection of P^test on D^train
    A_test_reg = np.linalg.inv(model_reg.D.T @ model_reg.D) @ model_reg.D.T @ P_test  # with reg
    A_test_noreg = np.linalg.inv(model_pca.D.T @ model_pca.D) @ model_pca.D.T @ P_test  # without reg

    # predictions reg
    # W = np.minimum(model_reg.W, 1 - 1e-10)  # clip to avoid w_k=1
    mu = model_reg.mu.reshape(-1, 1)
    A_pred_reg = mu + (model_reg.W.reshape(-1, 1) * A_test_reg[:, :-1])  # without the last testing value

    # predictions no-reg
    list_paramsAR = []  # (constant, beta_1)
    for atom in range(K):
        ar_model = AutoReg(A_test_noreg[atom, :], lags=1).fit()  # AR1 model
        list_paramsAR.append(ar_model.params)
    mat_paramsAR = np.array(list_paramsAR)
    A_pred_noreg = mat_paramsAR[:, 0].reshape(-1, 1) + mat_paramsAR[:, 1].reshape(-1, 1) * A_test_noreg[:, :-1]

    # return model_pca.D, A_test_noreg, A_pred_noreg, model_reg.D, A_test_reg, A_pred_reg
    return model_pca.D, A_pred_noreg, model_reg.D, A_pred_reg


def cross_lambda_training(P, K, r_dim, list_lambdas):
    """
    DL training for several lambdas
    Parameters
    ----------
    P : ndarray
        migration matrices
    K: int
        number of atoms
    r_dim: int
        matrix dimenion


    Returns
    -------
    dict_reco:  dict
    dict_regu:  dict
    dict_obj:  dict
    dict_codings:  dict
    """
    dict_reco = {}
    dict_regu = {}
    dict_obj = {}
    dict_codings = {}

    for lamb in list_lambdas:
        # fit
        model = DictionaryLearning(K=K, r_dim=r_dim)
        model.fit(P, lamb=lamb, max_iter=500)
        # save
        dict_reco[lamb] = model.list_reconstruction
        dict_regu[lamb] = model.list_regularization
        dict_obj[lamb] = model.list_total_objective
        dict_codings[lamb] = model.A
    return dict_reco, dict_regu, dict_obj, dict_codings