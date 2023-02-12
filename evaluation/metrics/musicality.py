import copy
import warnings

import numpy as np
from scipy.stats import entropy


def information_rate(m):
    """
    :param m: Guo matrix of a fragment
    """

    stats = np.zeros([74, 74])
    ir = []

    for t in range(1, m.shape[0]):
        try:
            stats[np.where(m[t - 1, :])[0][0], np.where(m[t, :])[0][0]] += 1
            ir.append(entropy(stats.sum(axis=0)) - entropy(stats[np.where(m[t - 1, :])[0][0], :]))
            # TODO: chequear si axis no es 1; da gráficos parecidos
        except:
            if len(list(np.where(m[t, :])[0])) == 0:
                warnings.warn(f"Skipping IR calculation of index {t} because it was empty.")
            else:
                warnings.warn(f"Skipping IR calculation of index {t} even if it was not empty.")
    return np.mean(ir)


def get_information_rate_table(df, inplace=True):
    """
    :param df: df with columns 'Title', 'Style', 'roll' and 'NewRoll'.
    :param inplace: whether to modify the input dataframe.
    :return: DataFrame with the same columns as input with the addition of 'IR orig' and 'IR trans'.
    """
    df_out = df if inplace else copy.copy(df)

    df_out["IR orig"] = df_out.apply(lambda row: information_rate(row["roll"].matrix), axis=1)
    df_out["IR trans"] = df_out.apply(lambda row: information_rate(row["NewRoll"].matrix), axis=1)

    return df_out
