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

    return A_test_noreg, A_pred_noreg, A_test_reg, A_pred_reg