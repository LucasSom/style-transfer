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
        "ms": cmp_interval_matrices(m_orig, orig_avg),
        "ms'": cmp_interval_matrices(m_orig, trans_avg),
        "m's": cmp_interval_matrices(m_trans, orig_avg),
        "m's'": cmp_interval_matrices(m_trans, trans_avg)
    }


def evaluate_interval_distribution(m_orig, m_trans, orig_avg, trans_avg):
    cmp = get_comparisons(m_orig, m_trans, orig_avg, trans_avg)
    return cmp["ms"] < cmp["ms'"] and cmp["ms"] < cmp["m's"], \
           cmp["m's'"] > cmp["ms'"] and cmp["m's'"] > cmp["m's"]


def get_interval_distances_table(df, orig=None, dest=None):
    """df_transferred"""
    orig_style_mx = get_style_avg(df, orig)
    orig_style_mx_norm = normalize(orig_style_mx)

    trans_style_mx = get_style_avg(df, dest)
    trans_style_mx_norm = normalize(trans_style_mx)

    table = {"Style": [],
             "Title": [],
             "target": [],
             "ms": [],
             "ms'": [],
             "m's": [],
             "m's'": [],
             "log(m's/ms)": [],
             "log(m's'/ms')": []
             }

    for title, style, r_orig, r_trans in zip(df["Title"], df[df["Style"] == orig]["Style"], df['roll'], df["Transferred"]):
        distances = get_comparisons(
            matrix_of_adjacent_intervals(r_orig)[0],
            matrix_of_adjacent_intervals(r_trans)[0],
            orig_style_mx_norm,
            trans_style_mx_norm)

        table["Style"].append(style)
        table["Title"].append(title)
        table["target"].append(dest)
        table["ms"].append(distances["ms"])
        table["ms'"].append(distances["ms'"])
        table["m's"].append(distances["m's"])
        table["m's'"].append(distances["m's'"])
        table["log(m's/ms)"].append(np.log(distances["m's"] / distances["ms"]))
        table["log(m's'/ms')"].append(np.log(distances["m's'"] / distances["ms'"]))

    return pd.DataFrame(table)
