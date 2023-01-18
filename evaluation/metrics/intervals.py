from typing import List

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from evaluation.metrics.metrics import get_matrix_comparisons
from utils.utils import normalize


def matrix_of_adjacent_intervals(roll_or_song, voice='melody'):
    intervals: List[int] = roll_or_song.get_adjacent_intervals(voice)

    return np.histogram2d(intervals[:-1], intervals[1:], bins=(range(-12, 13), range(-12, 13)))


def plot_matrix_of_adjacent_intervals(song, voice='melody'):
    intervals = song.get_adjacent_intervals(voice)

    p = plt.hist2d(intervals[:-1], intervals[1:])
    plt.title(f"{song.name}-{voice}-intervals")
    return p


def get_interval_distribution_params(intervals: List[int]):
    intervals = np.array(intervals)
    return intervals.mean(), intervals.std()


def get_style_intervals_bigrams_avg(df: pd.DataFrame, style: str) -> np.array:
    """
    Compute the average distribution of the interval bigrams of the rolls of the style
    :param df: df with columns 'Style' and 'roll'
    :param style: it must be one of the Style column

    :return: matrix of 24x24 with the average distribution of bigrams of musical intervals for the style
    """
    avg = np.zeros((24, 24))
    df_style = df[df['Style'] == style]

    for roll in df_style['roll']:
        avg += matrix_of_adjacent_intervals(roll)[0]

    assert df_style.shape[0] != 0
    return normalize(avg / df_style.shape[0])


def evaluate_interval_distribution(m_orig, m_trans, orig_avg, trans_avg):
    cmp = get_matrix_comparisons(m_orig, m_trans, orig_avg, trans_avg)
    return cmp["ms"] < cmp["ms'"] and cmp["ms"] < cmp["m's"], \
           cmp["m's'"] > cmp["ms'"] and cmp["m's'"] > cmp["m's"]


