import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from typing import Union

from roll.guoroll import GuoRoll
from roll.song import Song

possible_patterns = 2 ** 4


def pattern_to_int(pattern):
    return int(pattern[0])*2**3 + int(pattern[1])*2**2 + int(pattern[2])*2**1 + int(pattern[3])*2**0


def matrix_of_adjacent_rhythmic_bigrams(roll_or_song: Union[GuoRoll, Song], voice='melody'):
    rps = list(map(pattern_to_int, roll_or_song.get_adjacent_rhythmic_patterns(voice)))

    # patterns = [np.zeros(possible_patterns, dtype=int) for _ in range(possible_patterns)]
    # for prev, nxt in zip(rps[:-1], rps[1:]):
    #     patterns[prev][nxt] += 1

    return np.histogram2d(rps[:-1], rps[1:], bins=(range(possible_patterns + 1), range(possible_patterns + 1)))


def plot_matrix_of_adjacent_rhythmic_bigrams(song: Song, voice='melody'):
    patterns = matrix_of_adjacent_rhythmic_bigrams(song, voice)

    plt.title(f"{song.name}-{voice}-rhythmic_patterns")

    return sns.heatmap(patterns, cmap='Oranges', annot=True, fmt='d')


def get_style_rhythmic_bigrams_avg(df: pd.DataFrame, style: str) -> np.array:
    """
    Compute the average distribution of the rhythmic bigrams of the rolls of the style
    :param df: df with columns 'Style' and 'roll'
    :param style: it must be one of the Style column

    :return: matrix of 16x16 with the average distribution of bigrams of rhythmic patterns for the style
    """
    avg = np.ones((16, 16))
    df_style = df[df['Style'] == style]

    for roll in df_style['roll']:
        avg += matrix_of_adjacent_rhythmic_bigrams(roll)[0]

    assert df_style.shape[0] != 0
    return avg / df_style.shape[0]
