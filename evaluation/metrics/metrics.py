import numpy as np
import pandas as pd
from pandas import DataFrame
from scipy.stats import entropy

from evaluation.metrics.intervals import matrix_of_adjacent_intervals, get_style_intervals_bigrams_avg
from evaluation.metrics.musicality import get_information_rate_table
from evaluation.metrics.plagiarism import get_plagiarism_ranking_table
from evaluation.metrics.rhythmic_bigrams import matrix_of_adjacent_rhythmic_bigrams, get_style_rhythmic_bigrams_avg
from utils.utils import get_matrix_comparisons


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


def obtain_metrics(df, e_orig, e_dest, characteristics, *argv):
    d = {"original_style": e_orig, "target_style": e_dest}
    for metric in argv:
        if metric == "rhythmic_bigrams":
            dist = get_distribution_distances(df, e_orig, e_dest, characteristics, rhythm=True)
            d["rhythmic_bigrams"] = dist

        if metric == "intervals":
            dist = get_distribution_distances(df, e_orig, e_dest, characteristics)
            d["intervals"] = dist

        if metric == "plagiarism": d["plagiarism"] = get_plagiarism_ranking_table(df)

        if metric == "musicality": d["musicality"] = get_information_rate_table(df)
    return d


def get_distribution_distances(df: pd.DataFrame, orig: str, dest: str, styles: dict, rhythm=False):
    """
    :param df: df with columns 'Style' and 'roll'
    :param orig: style of rolls to transform
    :param dest: style destiny to transform
    :param styles: dictionary of styles characteristics
    :param rhythm: whether to compute the distances of rhythmic intervals bigrams
    :return: dataframe of input with new columns with the logarithmic distances to the characteristic interval and rhythmic distributions of original and new style
    """
    orig_style_mx_norm = styles[orig].rhythmic_bigrams_distribution if rhythm else styles[orig].intervals_distribution
    trans_style_mx_norm = styles[dest].rhythmic_bigrams_distribution if rhythm else styles[dest].intervals_distribution

    table = {"Style": [],
             "Title": [],
             "roll": [],
             "NewRoll": [],
             "target": [],
             "m": [],
             "m'": [],
             "ms": [],
             "ms'": [],
             "m's": [],
             "m's'": [],
             "log(m's/ms)": [],
             "log(m's'/ms')": []
             }

    for s1, s2 in [(orig, dest), (dest, orig)]:
        sub_df = df[df["Style"] == s1]
        for title, r_orig, r_trans in zip(sub_df["Title"], sub_df['roll'], sub_df["NewRoll"]):
            m_orig = matrix_of_adjacent_rhythmic_bigrams(r_orig)[0] if rhythm else matrix_of_adjacent_intervals(r_orig)[0]
            m_trans = matrix_of_adjacent_rhythmic_bigrams(r_trans)[0] if rhythm else matrix_of_adjacent_intervals(r_trans)[0]
            distances = get_matrix_comparisons(m_orig, m_trans, orig_style_mx_norm, trans_style_mx_norm)

            table["Style"].append(s1)
            table["Title"].append(title)
            table["roll"].append(r_orig)
            table["NewRoll"].append(r_trans)
            table["target"].append(s2)
            table["m"].append(m_orig)
            table["m'"].append(m_trans)
            table["ms"].append(distances["ms"])
            table["ms'"].append(distances["ms'"])
            table["m's"].append(distances["m's"])
            table["m's'"].append(distances["m's'"])
            table["log(m's/ms)"].append(np.log(distances["m's"] / distances["ms"]))
            table["log(m's'/ms')"].append(np.log(distances["m's'"] / distances["ms'"]))

    return pd.DataFrame(table)
