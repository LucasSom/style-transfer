import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from typing import Union

from model.colab_tension_vae import params
from roll.guoroll import GuoRoll, get_rp
from roll.song import Song
from utils.utils import normalize

possible_patterns = 2 ** 4


def pattern_to_int(pattern):
    return int(pattern[0])*2**3 + int(pattern[1])*2**2 + int(pattern[2])*2**1 + int(pattern[3])*2**0


def matrix_of_adjacent_rhythmic_bigrams(roll_or_matrix: Union[GuoRoll, np.ndarray, Song], voice='melody'):
    if type(roll_or_matrix) is np.ndarray:
        rps = list(map(pattern_to_int, get_rp(roll_or_matrix[:, params.config.melody_dim])))
    else:
        rps = list(map(pattern_to_int, roll_or_matrix.get_adjacent_rhythmic_patterns(voice)))

    # patterns = [np.zeros(possible_patterns, dtype=int) for _ in range(possible_patterns)]
    # for prev, nxt in zip(rps[:-1], rps[1:]):
    #     patterns[prev][nxt] += 1

    return np.histogram2d(rps[:-1], rps[1:], bins=(range(possible_patterns + 1), range(possible_patterns + 1)))


def plot_matrix_of_adjacent_rhythmic_bigrams(song: Song, voice='melody'):
    patterns = matrix_of_adjacent_rhythmic_bigrams(song, voice)

    plt.title(f"{song.name}-{voice}-rhythmic_patterns")

    # return sns.heatmap(patterns[0], cmap='Oranges', annot=True, fmt='d')
    return sns.heatmap(patterns[0], cmap='Oranges', annot=True)


def get_rhythmic_distribution(df: pd.DataFrame) -> np.array:
    """
    Compute the average distribution of the rhythmic bigrams of the rolls of the style
    :param df: sub df with rolls of the style to which calculate de distribution.

    :return: matrix of 16x16 with the average distribution of bigrams of rhythmic patterns for the style
    """
    acc = np.ones((16, 16))

    acc = get_style_rhythmic_bigrams_sum(acc, df)[0]

    assert df.shape[0] != 0
    return normalize(acc / df.shape[0])


def get_style_rhythmic_bigrams_sum(acc, df_style):
    for roll in df_style['roll']:
        acc += matrix_of_adjacent_rhythmic_bigrams(roll)[0]
    return acc, range(possible_patterns + 1), range(possible_patterns + 1)
