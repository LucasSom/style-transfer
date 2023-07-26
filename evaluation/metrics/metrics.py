import pandas as pd

from evaluation.metrics.intervals import matrix_of_adjacent_intervals
from evaluation.metrics.musicality import get_information_rate_table
from evaluation.metrics.plagiarism import get_plagiarism_ranking_table
from evaluation.metrics.rhythmic_bigrams import matrix_of_adjacent_rhythmic_bigrams


def obtain_metrics(df, e_orig, e_dest, *argv):
    d = {"original_style": e_orig, "target_style": e_dest}
    for metric in argv:
        if metric == "rhythmic_bigrams":
            dist = get_distribution_distances(df, e_orig, e_dest, rhythm=True)
            d["rhythmic_bigrams"] = dist

        if metric == "intervals":
            dist = get_distribution_distances(df, e_orig, e_dest)
            d["intervals"] = dist

        if metric == "plagiarism":
            d["plagiarism-dist"] = get_plagiarism_ranking_table(df, e_orig, e_dest, by_distance=True)
            d["plagiarism-diff"] = get_plagiarism_ranking_table(df, e_orig, e_dest, by_distance=False)

        if metric == "musicality": d["musicality"] = get_information_rate_table(df)
    return d


def get_distribution_distances(df: pd.DataFrame, orig: str, dest: str, rhythm=False):
    """
    :param df: df with columns 'Style' and 'roll'
    :param orig: style of rolls to transform
    :param dest: style destiny to transform
    :param rhythm: whether to compute the distances of rhythmic intervals bigrams
    :return: dataframe of input with new columns with the logarithmic distances to the characteristic interval and rhythmic distributions of original and new style
    """
    table = {"Style": [],
             "Title": [],
             "roll": [],
             "NewRoll": [],
             "target": [],
             "m": [],
             "m'": [],
             }

    for s1, s2 in [(orig, dest), (dest, orig)]:
        sub_df = df[df["Style"] == s1]
        for title, r_orig, r_trans in zip(sub_df["Title"], sub_df['roll'], sub_df["NewRoll"]):
            m_orig = matrix_of_adjacent_rhythmic_bigrams(r_orig)[0] if rhythm else matrix_of_adjacent_intervals(r_orig)[0]
            m_trans = matrix_of_adjacent_rhythmic_bigrams(r_trans)[0] if rhythm else matrix_of_adjacent_intervals(r_trans)[0]

            table["Style"].append(s1)
            table["Title"].append(title)
            table["roll"].append(r_orig)
            table["NewRoll"].append(r_trans)
            table["target"].append(s2)
            table["m"].append(m_orig)
            table["m'"].append(m_trans)

    return pd.DataFrame(table)
