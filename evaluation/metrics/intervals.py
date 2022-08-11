import copy
from typing import List
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import entropy

from model.colab_tension_vae.params import config


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
	:param df: df_transferred
	:param style: it must be one of the Style column
	"""
    avg = np.zeros((25, 25))
    df_style = df[df['Style'] == style]

    for roll in df_style['roll']:
        avg += matrix_of_adjacent_intervals(roll)[0]

    return avg / df_style.shape[0]


def cmp_interval_matrices(m, avg):
    return np.mean([entropy(avg, m), entropy(m, avg)])


def get_comparisons(m_orig, m_trans, orig_avg, trans_avg):
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


def get_interval_distances_table(df, column, orig=None, dest=None, eps=0.00001):
    """df_transferred"""
    orig_style_mx = get_style_avg(df, orig)
    trans_style_mx = get_style_avg(df, dest)
    orig_style_mx_norm = orig_style_mx + eps / np.sum(orig_style_mx + eps)
    trans_style_mx_norm = trans_style_mx + eps / np.sum(trans_style_mx + eps)

    table = {"Title": [],
             "orig_name": [],
             "dest_name": [],
             "oo": [],
             "ot": [],
             "to": [],
             "tt": []
             }

    for title, r_orig, r_trans in zip(df["Title"], df['roll'], df[column]):
        distances = get_comparisons(r_orig.matrix, r_trans.matrix, orig_style_mx_norm, trans_style_mx_norm)
        table["Title"].append(title)
        table["oo"].append(distances["oo"])
        table["ot"].append(distances["ot"])
        table["to"].append(distances["to"])
        table["tt"].append(distances["tt"])

    return pd.DataFrame(table)
