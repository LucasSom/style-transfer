from typing import List

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import entropy

eps = 0.0001


def matrix_of_adjacent_intervals(roll_or_song, voice='melody'):
    intervals = roll_or_song.get_adjacent_intervals(voice)

    return np.histogram2d(intervals[:-1], intervals[1:], bins=(range(-12, 13), range(-12, 13)))


def plot_matrix_of_adjacent_intervals(song, voice='melody'):
    intervals = song.get_adjacent_intervals(voice)

    p = plt.hist2d(intervals[:-1], intervals[1:])
    plt.title(f"{song.name}-{voice}-intervals")
    return p


def get_interval_distribution_params(intervals: List[int]):
    intervals = np.array(intervals)
    return intervals.mean(), intervals.std()


def get_style_avg(df, style: str):
    """
    Computes the average embedding of the rolls of the style
    :param df: df_transferred
    :param style: it must be one of the Style column
    """
    avg = np.zeros((24, 24))
    df_style = df[df['Style'] == style]

    for roll in df_style['roll']:
        avg += matrix_of_adjacent_intervals(roll)[0]

    return avg / df_style.shape[0]


def normalize(m):
    m_sum = np.sum(m + eps)
    return m + eps / m_sum


def cmp_interval_matrices(m, avg):
    assert m.shape == avg.shape
    m_normalized = normalize(m)
    return np.mean([entropy(avg, m_normalized), entropy(m_normalized, avg)])


def get_comparisons(m_orig, m_trans, orig_avg, trans_avg):
    """
    :param m_orig: interval matrix of the original roll
    :param m_trans: interval matrix of the transformed roll
    :param orig_avg: interval matrix from the original style
    :param trans_avg: interval matrix from the target style
    """
    return {
        "oo": cmp_interval_matrices(m_orig, orig_avg),
        "ot": cmp_interval_matrices(m_orig, trans_avg),
        "to": cmp_interval_matrices(m_trans, orig_avg),
        "tt": cmp_interval_matrices(m_trans, trans_avg)
    }


def evaluate_interval_distribution(m_orig, m_trans, orig_avg, trans_avg):
    cmp = get_comparisons(m_orig, m_trans, orig_avg, trans_avg)
    return cmp["oo"] < cmp["ot"] and cmp["oo"] < cmp["to"], \
           cmp["tt"] > cmp["ot"] and cmp["tt"] > cmp["to"]


def get_interval_distances_table(df, orig=None, dest=None):
    """df_transferred"""
    orig_style_mx = get_style_avg(df, orig)
    orig_style_mx_norm = normalize(orig_style_mx)

    trans_style_mx = get_style_avg(df, dest)
    trans_style_mx_norm = normalize(trans_style_mx)

    table = {"Style": [],
             "Title": [],
             # "dest_name": [],
             "oo": [],
             "ot": [],
             "to": [],
             "tt": [],
             "log(tt/ot)": [],
             "log(ot/oo)": []
             }

    for title, style, r_orig, r_trans in zip(df["Title"], df["Style"], df['roll'], df["Transferred"]):
        distances = get_comparisons(
            matrix_of_adjacent_intervals(r_orig)[0],
            matrix_of_adjacent_intervals(r_trans)[0],
            orig_style_mx_norm,
            trans_style_mx_norm)
        table["Style"].append(style)
        table["Title"].append(title)
        table["oo"].append(distances["oo"])
        table["ot"].append(distances["ot"])
        table["to"].append(distances["to"])
        table["tt"].append(distances["tt"])
        table["log(tt/ot)"].append(np.log(distances["tt"] / distances["ot"]))
        table["log(ot/oo)"].append(np.log(distances["ot"] / distances["oo"]))

    return pd.DataFrame(table)
