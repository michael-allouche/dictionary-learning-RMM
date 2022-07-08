from models import DictionaryLearning

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
        reconstruction
    dict_regu:  dict
        regularization
    dict_obj:  dict
        objective
    dict_codings:  dict
        codings
    """
    dict_reco = {}
    dict_regu = {}
    dict_obj = {}
    dict_codings = {}

    for lamb in list_lambdas:
        # fit
        model = DictionaryLearning(K=K, r_dim=r_dim)
        model.fit(P, lamb=lamb, max_iter=500)
        # save serires
        dict_reco[lamb] = model.list_reconstruction
        dict_regu[lamb] = model.list_regularization
        dict_obj[lamb] = model.list_total_objective
        dict_codings[lamb] = model.A
    return dict_reco, dict_regu, dict_obj, dict_codings