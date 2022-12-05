import numpy as np
import pandas as pd
from scipy.stats import entropy

from model.colab_tension_vae.params import config


def information_rate(m):
    """
    :param m: matrix of
    """

    stats = np.zeros([74, 74])
    ir = []

    for t in range(1, m.shape[0]):
        stats[np.where(m[t - 1, :])[0][0], np.where(m[t, :])[0][0]] += 1
        ir.append(entropy(stats.sum(axis=0)) - entropy(stats[np.where(m[t - 1, :])[0][0], :]))
    return np.mean(ir)


def get_information_rate_table(df):
    """
    :param df: df_transferred
    """
    table = {"Style": [],
             "Title": [],
             # "dest_name": [],
             "IR orig": [],
             "IR trans": [],
             }

    for title, style, r_orig, r_trans in zip(df["Title"], df["Style"], df['roll'], df["Transferred"]):
        table["Style"].append(style)
        table["Title"].append(title)
        table["IR orig"].append(information_rate(r_orig.matrix))
        table["IR trans"].append(information_rate(r_trans.matrix))

    return pd.DataFrame(table)
