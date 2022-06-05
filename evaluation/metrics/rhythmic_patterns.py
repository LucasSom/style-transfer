import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

from roll.song import Song

possible_patterns = 2 ** 4


def pattern_to_int(pattern):
    return int(pattern[0])*2**3 + int(pattern[1])*2**2 + int(pattern[2])*2**1 + int(pattern[3])*2**0


def matrix_of_adjacent_rhythmic_patterns(song: Song, voice='melody'):
    rps = list(map(pattern_to_int, song.get_adjacent_rhythmic_patterns(voice)))

    patterns = [np.zeros(possible_patterns, dtype=int) for _ in range(possible_patterns)]
    for prev, nxt in zip(rps[:-1], rps[1:]):
        patterns[prev][nxt] += 1

    return patterns


def plot_matrix_of_adjacent_rhythmic_patterns(song: Song, voice='melody'):
    patterns = matrix_of_adjacent_rhythmic_patterns(song, voice)

    plt.title(f"{song.name}-{voice}-rhythmic_patterns")

    return sns.heatmap(patterns, cmap='Oranges', annot=True, fmt='d')
