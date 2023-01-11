import numpy as np
import pandas as pd

from evaluation.metrics.intervals import get_style_intervals_bigrams_avg, get_matrix_comparisons, matrix_of_adjacent_intervals
from evaluation.metrics.plagiarism import get_plagiarism_ranking_table
from evaluation.metrics.rhythmic_patterns import get_style_rhythmic_bigrams_avg, matrix_of_adjacent_rhythmic_bigrams


def obtain_metrics(df, e_orig, e_dest):
    return {"original_style": e_orig,
            "target_style": e_dest,
            "plagiarism": get_plagiarism_ranking_table(df),
            "intervals": get_distribution_distances_df(df, e_orig, e_dest),
            "rhythmic_bigrams": get_distribution_distances_df(df, e_orig, e_dest, rhythm=True),
            # "musicality": get_information_rate_table(df)
            }


def get_distribution_distances_df(df: pd.DataFrame, orig: str, dest: str, rhythm=False):
    """
    :param df: df with columns 'Style' and 'roll'
    :param orig: style of rolls to transform
    :param dest: style destiny to transform
    :param rhythm:
    :return: dataframe of input with new columns with the logarithmic distances to the characteristic interval and rhythmic distributions of original and new style
    """
    orig_style_mx_norm = get_style_rhythmic_bigrams_avg(df, orig) if rhythm else get_style_intervals_bigrams_avg(df, orig)
    trans_style_mx_norm = get_style_rhythmic_bigrams_avg(df, dest) if rhythm else get_style_intervals_bigrams_avg(df, dest)

    table = {"Style": [],
             "Title": [],
             "target": [],
             "ms": [],
             "ms'": [],
             "m's": [],
             "m's'": [],
             "log(m's/ms)": [],
             "log(m's'/ms')": []
             }

    sub_df = df[df["Style"] == orig]
    for title, style, r_orig, r_trans in zip(sub_df["Title"], sub_df["Style"], sub_df['roll'], sub_df["NewRoll"]):
        distances = get_matrix_comparisons(
            matrix_of_adjacent_rhythmic_bigrams(r_orig) if rhythm else matrix_of_adjacent_intervals(r_orig)[0],
            matrix_of_adjacent_rhythmic_bigrams(r_trans) if rhythm else matrix_of_adjacent_intervals(r_trans)[0],
            orig_style_mx_norm,
            trans_style_mx_norm)

        table["Style"].append(style)
        table["Title"].append(title)
        table["target"].append(dest)
        table["ms"].append(distances["ms"])
        table["ms'"].append(distances["ms'"])
        table["m's"].append(distances["m's"])
        table["m's'"].append(distances["m's'"])
        table["log(m's/ms)"].append(np.log(distances["m's"] / distances["ms"]))
        table["log(m's'/ms')"].append(np.log(distances["m's'"] / distances["ms'"]))

    return pd.DataFrame(table)
