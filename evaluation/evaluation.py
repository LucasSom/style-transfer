import os

import dfply
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from typing import List

from evaluation.metrics.intervals import get_interval_distances_table
from evaluation.metrics.plagiarism import sort_by_general_plagiarism, get_most_similar_roll
from utils.files_utils import data_path


def evaluate_model(df, metrics, column=None):
    print("===== Evaluate interval distributions =====")
    o, t = get_interval_distances_table(df)
    print("How many rolls moved away?", o)
    print("How many rolls moved closer to the new style?", t)


def evaluate_plagiarism_coincidences(df, direction) -> float:
    rolls = list(df['rolls'])
    base_rolls = df[direction]
    titles = list(df['Title'])

    similarities = [title == get_most_similar_roll(base_roll, rolls).song.name
                    for title, base_roll in zip(titles, base_rolls)]
    return sum(similarities) / len(similarities)


def evaluate_plagiarism_rate(df, direction) -> (float, float):
    rolls = list(df['rolls'])
    titles = list(df['Title'])
    base_rolls = df[direction]

    distincts = 0
    for title, base_roll in zip(titles, base_rolls):
        sorted_rolls = sort_by_general_plagiarism(rolls, base_roll)
        for r in sorted_rolls:
            if r.song.name == title:
                break
            else:
                distincts += 1
    return distincts, len(rolls)


def evaluate_single_intervals_distribution(df, orig, dest):
    distances_df = get_interval_distances_table(df, orig, dest)

    sns.set_theme()
    sns.kdeplot(data=distances_df, x="log(tt/ot)")
    sns.displot(data=distances_df, x="log(ot/oo)", kind="kde")
    plt.title(f'Interval distribution of {orig} transformed to {dest}')
    plt.show()
    return distances_df


def evaluate_multiple_intervals_distribution(dfs: List[pd.DataFrame]):
    """
    Estos dfs provendrían de cada df de ida y vuelta. Es decir, serían 6 dfs distintos.
    Considerando esto, en cada df voy a tener 2 estilos, así que evalúo single con ambos.
    """
    merged_df = pd.DataFrame()
    for df in dfs:
        s1 = list(set(df["Style"]))[0]
        s2 = list(set(df["Style"]))[1]

        df1 = evaluate_single_intervals_distribution(df, s1, s2)
        df1["orig"] = [s1 for _ in range(df1.shape[0])]
        df1["target"] = [s2 for _ in range(df1.shape[0])]

        df2 = evaluate_single_intervals_distribution(df, s2, s1)
        df2["orig"] = [s2 for _ in range(df2.shape[0])]
        df2["target"] = [s1 for _ in range(df2.shape[0])]

        merged_df = pd.concat([merged_df, df1, df2])

    merged_df = merged_df >> dfply.gather("type", "value", ["log(tt/ot)", "log(ot/oo)"])

    sns.displot(data=merged_df, col="target", row="orig", x="value", hue="type", kind="kde")
    plt.savefig(os.path.join(data_path, "debug_outputs", "intervals_plot.png"))
    plt.show()
