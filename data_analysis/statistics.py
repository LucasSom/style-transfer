import numpy as np
import pandas as pd
from pandas import DataFrame
from scipy.stats import entropy
from sklearn.model_selection import StratifiedShuffleSplit

from data_analysis.assemble_data import optimal_transport
from evaluation.metrics.intervals import get_intervals_distribution
from evaluation.metrics.rhythmic_bigrams import get_rhythmic_distribution


def styles_bigrams_entropy(df) -> DataFrame:
    def calculate_entropy(df_style, interval):
        probabilities = get_intervals_distribution(df_style) if interval else get_rhythmic_distribution(df_style)
        # return entropy(probabilities, axis=0)
        return entropy(np.hstack(probabilities))
        # TODO: Consultar a March. Lo que quiero es la entropía de la entropía o la entropía de stackear las probabilidades?

    d = {"Style": [], "Melodic entropy": [], "Rhythmic entropy": []}
    for style in set(df["Style"]):
        d["Style"].append(style)
        d["Melodic entropy"].append(calculate_entropy(df[df["Style"] == style], True))
        d["Rhythmic entropy"].append(calculate_entropy(df[df["Style"] == style], False))

    return pd.DataFrame(d)


def stratified_split(df, n_splits=1):
    strat_split = StratifiedShuffleSplit(n_splits=n_splits, test_size=0.2, random_state=42)

    train_dfs = [df.loc[train_index] for train_index, _ in strat_split.split(df, df["Style"])]
    test_dfs = [df.loc[test_index] for _, test_index in strat_split.split(df, df["Style"])]

    return train_dfs, test_dfs


def style_ot_diff(s1, s2, histograms, melodic=True):
    if s1 != s2:
        h1 = histograms[s1]["melodic_hist" if melodic else "rhythmic_hist"]
        h2 = histograms[s2]["melodic_hist" if melodic else "rhythmic_hist"]

        return optimal_transport(h1, h2, melodic)
    return 0


def styles_ot_table(df, histograms):
    diff_table = pd.DataFrame(
        {
            's1': s1,
            's2': s2,
            'd': style_ot_diff(s1, s2, histograms)
        }
        for s1 in set(df['Style'])
        for s2 in set(df['Style'])
    )
    return diff_table


def closest_ot_style(df, histograms):
    d = {}
    for i, s in enumerate(set(df['Style'])):
        style_melodic_hist = histograms[s]["melodic_hist"]
        style_rhythmic_hist = histograms[s]["rhythmic_hist"]

        df[f'Melodic ot to {s}'] = df.apply(lambda row: optimal_transport(
            row["Melodic bigram matrix"],
            style_melodic_hist, melodic=True), axis=1)

        df[f'Rhythmic ot to {s}'] = df.apply(lambda row: optimal_transport(
            row["Rhythmic bigram matrix"],
            style_rhythmic_hist, melodic=False), axis=1)

        df[f'Joined ot to {s}'] = df.apply(lambda row: row[f'Melodic ot to {s}'] + row[f'Rhythmic ot to {s}'], axis=1)

        d[i] = s

    for kind in ["Melodic", "Rhythmic", "Joined"]:
        df[f"{kind} closest style (ot)"] = df[[f'{kind} ot to {s}' for s in set(df['Style'])]].apply(np.argmin, axis=1)
        df[f"{kind} closest style (ot)"] = df.apply(lambda row: d[row[f"{kind} closest style (ot)"]], axis=1)

    return df