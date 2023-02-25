import numpy as np
import pandas as pd
from pandas import DataFrame
from scipy.stats import entropy
from sklearn.model_selection import StratifiedShuffleSplit

from data_analysis.assemble_data import optimal_transport
from evaluation.metrics.intervals import get_style_intervals_bigrams_avg
from evaluation.metrics.rhythmic_bigrams import get_style_rhythmic_bigrams_avg


def styles_bigrams_entropy(df) -> DataFrame:
    def calculate_entropy(df_style, style, interval):
        probabilities = get_style_intervals_bigrams_avg(df_style, style) if interval else get_style_rhythmic_bigrams_avg(df_style, style)
        # return entropy(probabilities, axis=0)
        return entropy(np.hstack(probabilities))
        # TODO: Consultar a March. Lo que quiero es la entropía de la entropía o la entropía de stackear las probabilidades?

    d = {"Style": [], "Melodic entropy": [], "Rhythmic entropy": []}
    for style in set(df["Style"]):
        d["Style"].append(style)
        d["Melodic entropy"].append(calculate_entropy(df[df["Style"] == style], style, True))
        d["Rhythmic entropy"].append(calculate_entropy(df[df["Style"] == style], style, False))

    return pd.DataFrame(d)


def stratified_split(df):
    strat_split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in strat_split.split(df, df["Style"]):
        strat_train_df = df.loc[train_index]
        strat_test_df = df.loc[test_index]

    return strat_train_df, strat_test_df


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