import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import date

# load monthly dates
sdate = date(2004, 1, 1)   # start date
edate = date(2020, 1, 1)   # end date

dates = pd.date_range(sdate,edate, freq='m')


def plot_rro(dict_reco, dict_regu, dict_obj, trunc=0):
    """plot reconstruction, regularization, objective with respect to the DL iterations"""

    fig, axes = plt.subplots(3, 1, figsize=(15, 13), sharex=False, squeeze=False)
    iterations = np.arange(len(list(dict_reco.values())[0]))[trunc:]

    for k, v in dict_reco.items():
        axes[0, 0].plot(iterations, dict_reco[k][trunc:], label=k)
        axes[0, 0].set_title(r"Reconstruction $||P-DA||_F^2$")
        # axes[0, 0].legend(title='Lambda', fontsize=15, title_fontsize=20)
        axes[0, 0].legend(title='Lambda', title_fontsize=15,
                          prop={"size": 13}, loc='upper center', bbox_to_anchor=(0.5, 1.5),
                          fancybox=True, shadow=True, ncol=4)
        # axes[0, 0].set_ylim(0.64, 0.85)

        axes[1, 0].plot(iterations, dict_regu[k][trunc:])
        axes[1, 0].set_title(
            r"Regularizaton $\sum_{k=1}^K\sum_{t =1}^{T-1} \left(\alpha_{k}^{t+1} - \bar\alpha_k - w_k(\alpha_{k}^{t} - \bar\alpha_k)\right)^2$")

        axes[2, 0].plot(iterations, dict_obj[k][trunc:])
        axes[2, 0].set_title(
            r"Objective $||P-DA||_F^2 + \lambda\sum_{k=1}^K\sum_{t =1}^{T-1} \left(\alpha_{k}^{t+1} - \bar\alpha_k - w_k(\alpha_{k}^{t} - \bar\alpha_k)\right)^2$")
        axes[2, 0].set_xlabel("DL iterations")

    for i in range(3):
        axes[i, 0].spines["left"].set_color("black")
        axes[i, 0].spines["bottom"].set_color("black")

    return


def plot_codings_lamb(dict_codings, atom):
    """
    plot the codings with repect tot he iterations for different values of lambda
    """

    fig, axes = plt.subplots(1, 1, figsize=(15, 7), sharex=False, squeeze=False)
    n_data = list(dict_codings.values())[0].shape[1]

    list_linestyle = ["-", "--", "-.", ":"]
    i = 0
    for k, v in dict_codings.items():
        # print(dates[n_train].shape, dict_codings[k][atom].shape)
        axes[0, 0].plot(dates[:n_data], dict_codings[k][atom-1], label=k, linewidth=2, linestyle=list_linestyle[i])
        axes[0, 0].set_title(f"Atom {atom}")
        axes[0, 0].legend(title='Lambda', title_fontsize=15,
                          prop={"size": 13}, loc='upper center', bbox_to_anchor=(0.5, 1.18),
                          fancybox=True, shadow=True, ncol=4)
        i += 1

    axes[0, 0].spines["left"].set_color("black")
    axes[0, 0].spines["bottom"].set_color("black")

    _ = plt.xticks(fontsize=20, rotation=30)
    _ = plt.yticks(fontsize=20)

    return
