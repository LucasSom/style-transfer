import numpy as np
import pandas as pd
from pandas import DataFrame
from scipy.stats import entropy, kstest
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


def stratified_split(df, n_splits=1, test_size=0.2):
    strat_split = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=42)

    df = df.reset_index()
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
    """
    For each roll it calculates the optimal transport distance to each style

    :param df: Dataframe with columns 'Style' 'Melodic bigram matrix' and 'Rhythmic bigram matrix'
    :param histograms: dictionary of melodic and rhythmic histograms of styles
    :return: Dataframe with new columns: for each style, s 'Melodic ot to {s}' and for each kind (Melodic, Rhythmic and
    Joined), '{kind} closest style (ot)'

    -------------
    Example:

    >> histograms = plot_styles_heatmaps_and_get_histograms(df, eval_dir)
    >> closest_ot_style(df, histograms)

    """
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


def test_musicality(df_train, df_test, df_permutations, melodic_distribution, rhythmic_distribution):
    df_test.name = "test"
    df_permutations.name = "permutations"
    for df in df_test, df_permutations:
        print(f"Size of dataframe {df.name}:", df.shape[0])
        melodic_test = kstest(df[f'Melodic musicality difference (probability)'],
                              df_train[f'Melodic musicality difference (probability)'])
        rhythmic_test = kstest(df[f'Rhythmic musicality difference (probability)'],
                               df_train[f'Rhythmic musicality difference (probability)'])

        melodic_test_less = kstest(df[f'Melodic musicality difference (probability)'],
                                   df_train[f'Melodic musicality difference (probability)'],
                                   alternative='less')
        rhythmic_test_less = kstest(df[f'Rhythmic musicality difference (probability)'],
                                    df_train[f'Rhythmic musicality difference (probability)'],
                                    alternative='less')

        melodic_test_greater = kstest(df[f'Melodic musicality difference (probability)'],
                                      df_train[f'Melodic musicality difference (probability)'],
                                      alternative='greater')
        rhythmic_test_greater = kstest(df[f'Rhythmic musicality difference (probability)'],
                                       df_train[f'Rhythmic musicality difference (probability)'],
                                       alternative='greater')

        print("Results of melodic kstest (equal):", melodic_test)
        print("Results of rhythmic kstest (equal):", rhythmic_test)
        print("Results of melodic kstest (less):", melodic_test_less)
        print("Results of rhythmic kstest (less):", rhythmic_test_less)
        print("Results of melodic kstest (greater):", melodic_test_greater)
        print("Results of rhythmic kstest (greater):", rhythmic_test_greater)

        test_equal = kstest(df[f'Joined musicality difference (probability)'],
                            df_train[f'Joined musicality difference (probability)'])

        test_less = kstest(df[f'Joined musicality difference (probability)'],
                           df_train[f'Joined musicality difference (probability)'],
                           alternative='less')

        test_greater = kstest(df[f'Joined musicality difference (probability)'],
                              df_train[f'Joined musicality difference (probability)'],
                              alternative='greater')

        print("Results of kstest (equal):", test_equal)
        print("Results of kstest (less):", test_less)
        print("Results of kstest (greater):", test_greater)
