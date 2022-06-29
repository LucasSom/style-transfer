from typing import List

import numpy as np
from matplotlib import pyplot as plt


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
