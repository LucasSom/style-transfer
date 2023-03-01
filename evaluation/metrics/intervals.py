from typing import List

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from utils.utils import normalize, get_matrix_comparisons


def matrix_of_adjacent_intervals(roll_or_song, voice='melody'):
    intervals: List[int] = roll_or_song.get_adjacent_intervals(voice)
    support = np.zeros((25, 25))
    for r, c in zip(intervals[:-1], intervals[1:]):
        if type(r) is not str and type(c) is not str:
            support[r+12,c+12] += 1
    return support, range(-12, 13), range(-12, 13)


def plot_matrix_of_adjacent_intervals(song, voice='melody'):
    intervals = song.get_adjacent_intervals(voice)

    p = plt.hist2d(intervals[:-1], intervals[1:])
    plt.title(f"{song.name}-{voice}-intervals")
    return p


def get_interval_distribution_params(intervals: List[int]):
    intervals = np.array(intervals)
    return intervals.mean(), intervals.std()


def get_intervals_distribution(df: pd.DataFrame) -> np.array:
    """
    Compute the average distribution of the interval bigrams of the rolls of the style
    :param df: sub df with rolls of the style to which calculate de distribution.

    :return: matrix of 25x25 with the average distribution of bigrams of musical intervals for the style
    """
    acc = np.zeros((25, 25))

    acc, _, _ = get_style_intervals_bigrams_sum(acc, df)

    assert df.shape[0] != 0
    return normalize(acc / df.shape[0])


def get_style_intervals_bigrams_sum(acc, df_style):
    for roll in df_style['roll']:
        acc += matrix_of_adjacent_intervals(roll)[0]
    return acc, range(-12, 13), range(-12, 13)


def evaluate_interval_distribution(m_orig, m_trans, orig_avg, trans_avg):
    cmp = get_matrix_comparisons(m_orig, m_trans, orig_avg, trans_avg)
    return cmp["ms"] < cmp["ms'"] and cmp["ms"] < cmp["m's"], \
           cmp["m's'"] > cmp["ms'"] and cmp["m's'"] > cmp["m's"]


