import numpy as np
from matplotlib import pyplot as plt

from utils.plots_utils import get_confusion_matrix


def matrix_of_adjacent_rhythmic_patterns(roll_or_song, voice='melody'):
    rps = roll_or_song.get_adjacent_rhythmic_patterns(voice)

    return np.histogram2d(rps[:-1], rps[1:])  # A chequear


def plot_matrix_of_adjacent_rhythmic_patterns(song, voice='melody'):
    rps = song.get_adjacent_rhythmic_patterns(voice)

    return get_confusion_matrix(rps, song, voice)
