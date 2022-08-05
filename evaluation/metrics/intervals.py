from typing import List
import numpy as np
from matplotlib import pyplot as plt

from model.colab_tension_vae.params import config


def matrix_of_adjacent_intervals(roll_or_song, voice='melody'):
    intervals = roll_or_song.get_adjacent_intervals(voice)

    return np.histogram2d(intervals[:-1], intervals[1:], bins=(range(-12, 13), range(-12, 13)))


def plot_matrix_of_adjacent_intervals(song, voice='melody'):
    intervals = song.get_adjacent_intervals(voice)

    p = plt.hist2d(intervals[:-1], intervals[1:])
    plt.title(f"{song.name}-{voice}-intervals")
    return p


def get_interval_distribution(intervals: List[int]):
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
