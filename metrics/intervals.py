import numpy as np
from matplotlib import pyplot as plt


def matrix_of_adjacent_intervals(roll_or_song, voice='melody'):
    intervals = roll_or_song.get_adjacent_intervals(voice)

    return np.histogram2d(intervals[:-1], intervals[1:])


def plot_matrix_of_adjacent_intervals(roll_or_song, voice='melody'):
    intervals = roll_or_song.get_adjacent_intervals(voice)

    p = plt.hist2d(intervals[:-1], intervals[1:])
    plt.show()
    return p
