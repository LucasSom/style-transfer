import numpy as np
import pandas as pd
from pandas import DataFrame
from scipy.special import rel_entr
from scipy.stats import entropy
from sklearn.model_selection import StratifiedShuffleSplit

from data_analysis.plots import plot_closeness
from evaluation.metrics.intervals import get_style_intervals_bigrams_avg, matrix_of_adjacent_intervals
from evaluation.metrics.rhythmic_bigrams import get_style_rhythmic_bigrams_avg, matrix_of_adjacent_rhythmic_bigrams
from model.embeddings.style import Style
from utils.utils import normalize


def styles_bigrams_entropy(df) -> DataFrame:
    def calculate_entropy(df_style, style, interval):
        probabilities = get_style_intervals_bigrams_avg(df_style, style) if interval else get_style_rhythmic_bigrams_avg(df_style, style)
        return entropy(probabilities, axis=0)

    d = {"Style": [], "Melodic entropy": [], "Rhythmic entropy": []}
    for style in set(df["Style"]):
        d["Style"].append(style)
        d["Melodic entropy"].append(calculate_entropy(df[df["Style"] == style], style, True))
        d["Rhythmic entropy"].append(calculate_entropy(df[df["Style"] == style], style, False))

    return pd.DataFrame(d)


def validate_style_belonging(df, eval_path, context='talk'):
    strat_split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in strat_split.split(df, df["Style"]):
        strat_train_df = df.loc[train_index]
        strat_test_df = df.loc[test_index]

    styles_names = set(strat_train_df["Style"])
    styles_train = {name: Style(name, None, strat_train_df) for name in styles_names}

    strat_test_df["Melodic bigram matrix"] = strat_test_df.apply(
        lambda row: matrix_of_adjacent_intervals(row["roll"])[0], axis=1)
    strat_test_df["Rhythmic bigram matrix"] = strat_test_df.apply(
        lambda row: matrix_of_adjacent_rhythmic_bigrams(row["roll"])[0], axis=1)

    strat_test_df["Rhythmic closest style (linear)"] = strat_test_df.apply(
        lambda row: rhythmic_closest_style(row["Rhythmic bigram matrix"], styles_train), axis=1)
    strat_test_df["Rhythmic closest style (kl)"] = strat_test_df.apply(
        lambda row: rhythmic_closest_style(row["Rhythmic bigram matrix"], styles_train, method='kl'), axis=1)
    strat_test_df["Rhythmic closest style (probability)"] = strat_test_df.apply(
        lambda row: rhythmic_closest_style(row["Rhythmic bigram matrix"], styles_train, method='probability'), axis=1)

    strat_test_df["Melodic closest style (linear)"] = strat_test_df.apply(
        lambda row: melodic_closest_style(row["Melodic bigram matrix"], styles_train), axis=1)
    strat_test_df["Melodic closest style (kl)"] = strat_test_df.apply(
        lambda row: melodic_closest_style(row["Melodic bigram matrix"], styles_train, method='kl'), axis=1)
    strat_test_df["Melodic closest style (probability)"] = strat_test_df.apply(
        lambda row: melodic_closest_style(row["Melodic bigram matrix"], styles_train, method='probability'), axis=1)

    strat_test_df["Joined closest style (linear)"] = strat_test_df.apply(
        lambda row: joined_closest_style(row["Melodic bigram matrix"], row["Rhythmic bigram matrix"], styles_train), axis=1)
    strat_test_df["Joined closest style (kl)"] = strat_test_df.apply(
        lambda row: joined_closest_style(row["Melodic bigram matrix"], row["Rhythmic bigram matrix"], styles_train, method='kl'), axis=1)
    strat_test_df["Joined closest style (probability)"] = strat_test_df.apply(
        lambda row: joined_closest_style(row["Melodic bigram matrix"], row["Rhythmic bigram matrix"], styles_train, method='probability'), axis=1)


    for orig in styles_names:
        plot_closeness(strat_test_df[strat_test_df["Style"] == orig], strat_test_df[strat_test_df["Style"] == orig],
                       orig, "nothing", eval_path + "/styles", context)


def kl(P, Q):
    return sum(sum(rel_entr(normalize(P), Q)))

def linear_distance(P, Q):
    return sum(sum(abs(normalize(P) - Q)))

def belonging_probability(M, x):
    return sum(sum(np.log(M) * x))

def rhythmic_closest_style(bigram_matrix, styles, method='linear'):
    if method == 'kl':
        min_style_idx = np.argmin(
            [kl(style.rhythmic_bigrams_distribution, bigram_matrix) for style in styles.values()]
        )
    elif method == 'linear':
        min_style_idx = np.argmin(
            [linear_distance(style.rhythmic_bigrams_distribution, bigram_matrix) for style in styles.values()]
        )
    elif method == 'probability':
        min_style_idx = np.argmin(
            [belonging_probability(style.rhythmic_bigrams_distribution, bigram_matrix) for style in styles.values()]
        )
    else:
        raise ValueError(f"{method} is not a valid method.")
    return list(styles.items())[min_style_idx][0]


def melodic_closest_style(bigram_matrix, styles, method='linear'):
    if method == 'kl':
        min_style_idx = np.argmin(
            [kl(style.intervals_distribution, bigram_matrix) for style in styles.values()]
        )
    elif method == 'linear':
        min_style_idx = np.argmin(
            [linear_distance(style.intervals_distribution, bigram_matrix) for style in styles.values()]
        )
    elif method == 'probability':
        min_style_idx = np.argmin(
            [belonging_probability(style.intervals_distribution, bigram_matrix) for style in styles.values()]
        )
    else:
        raise ValueError(f"{method} is not a valid method.")
    return list(styles.items())[min_style_idx][0]


def joined_closest_style(interval_matrix, rhythmic_matrix, styles, method='linear'):
    if method == 'kl':
        min_style_idx = np.argmin(
            [kl(style.intervals_distribution, interval_matrix)
             + kl(style.rhythmic_bigrams_distribution, rhythmic_matrix)
             for style in styles.values()]
        )
    elif method == 'linear':
        min_style_idx = np.argmin(
            [linear_distance(style.intervals_distribution, interval_matrix)
             + linear_distance(style.rhythmic_bigrams_distribution, rhythmic_matrix)
             for style in styles.values()]
        )
    elif method == 'probability':
        min_style_idx = np.argmin(
            [belonging_probability(style.intervals_distribution, interval_matrix)
             + belonging_probability(style.rhythmic_bigrams_distribution, rhythmic_matrix)
             for style in styles.values()]
        )
    else:
        raise ValueError(f"{method} is not a valid method.")
    return list(styles.items())[min_style_idx][0]