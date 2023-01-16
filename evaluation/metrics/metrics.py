import numpy as np
import pandas as pd

from evaluation.metrics.intervals import get_matrix_comparisons, matrix_of_adjacent_intervals
from evaluation.metrics.plagiarism import get_plagiarism_ranking_table
from evaluation.metrics.rhythmic_bigrams import matrix_of_adjacent_rhythmic_bigrams


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

        # if metric == "musicality": d["intervals"] = get_information_rate_table(df)
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

    sub_df = df[df["Style"] == orig]
    for title, style, r_orig, r_trans in zip(sub_df["Title"], sub_df["Style"], sub_df['roll'], sub_df["NewRoll"]):
        m_orig = matrix_of_adjacent_rhythmic_bigrams(r_orig) if rhythm else matrix_of_adjacent_intervals(r_orig)[0]
        m_trans = matrix_of_adjacent_rhythmic_bigrams(r_trans) if rhythm else matrix_of_adjacent_intervals(r_trans)[0]
        distances = get_matrix_comparisons(m_orig, m_trans, orig_style_mx_norm, trans_style_mx_norm)

        table["Style"].append(style)
        table["Title"].append(title)
        table["target"].append(dest)
        table["m"].append(m_orig)
        table["m'"].append(m_trans)
        table["ms"].append(distances["ms"])
        table["ms'"].append(distances["ms'"])
        table["m's"].append(distances["m's"])
        table["m's'"].append(distances["m's'"])
        table["log(m's/ms)"].append(np.log(distances["m's"] / distances["ms"]))
        table["log(m's'/ms')"].append(np.log(distances["m's'"] / distances["ms'"]))

    return pd.DataFrame(table)
