import os
from typing import List

import dfply
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import PercentFormatter

from evaluation.metrics.intervals import get_interval_distances_table
from evaluation.metrics.plagiarism import get_most_similar_roll, get_plagiarism_ranking_table
from model.colab_tension_vae.params import init
from utils.files_utils import data_path, datasets_debug_path, load_pickle
from utils.plots_utils import intervals_plot


def evaluate_model(df, metrics, column=None):
    print("===== Evaluate interval distributions =====")
    o, t = get_interval_distances_table(df)
    print("How many rolls moved away from the original style?", o)
    print("How many rolls moved closer to the new style?", t)


def evaluate_plagiarism_coincidences(df, direction, by_distance=False) -> float:
    rolls = list(df['rolls'])
    base_rolls = df[direction]
    titles = list(df['Title'])

    similarities = [title == get_most_similar_roll(base_roll, rolls, by_distance).song.name
                    for title, base_roll in zip(titles, base_rolls)]
    return sum(similarities) / len(similarities)


def evaluate_single_plagiarism(df, orig, dest, plot=True, context='talk'):
    plagiarism_df = get_plagiarism_ranking_table(df)

    if plot:
        sns.set_theme()
        sns.set_context(context)
        sns.displot(data=plagiarism_df, x="Differences absolute ranking")
        sns.displot(data=plagiarism_df, x="Distance absolute ranking", kind="kde")
        plt.title(f'Plagiarism ranking of {orig} transformed to {dest}')
        plt.show()
    return plagiarism_df


def plagiarism_plot(df, order, context):
    if len(order) == 2:
        col = [order[0]]
        row = [order[1]]
        orig, dest = order
    else:
        col = row = order
        orig = dest = 'all'

    sns.set_theme()
    sns.set_context(context)

    sns.displot(data=df,
                col="target",
                row="orig",
                x="value",
                hue="type",
                kind='hist',
                stat='proportion',
                col_order=col,
                row_order=row)

    plt.gca().yaxis.set_major_formatter(PercentFormatter(1))

    plt.savefig(os.path.join(data_path, "debug_outputs", f"plagiarism_{orig}_to_{dest}.png"))
    plt.show()


def evaluate_multiple_plagiarism(dfs: List[pd.DataFrame], merge, context='talk'):
    """
    Estos dfs provendrían de cada df de ida y vuelta. Es decir, serían 6 dfs distintos.
    Considerando esto, en cada df voy a tener 2 estilos, así que evalúo single con ambos.
    """
    merged_df = pd.DataFrame()
    dfs_to_plot = []
    for df in dfs:
        s1 = list(set(df["Style"]))[0]
        s2 = list(set(df["Style"]))[1]

        df1 = evaluate_single_plagiarism(df, s1, s2, False, context)
        df1["orig"] = [s1 for _ in range(df1.shape[0])]
        df1["target"] = [s2 for _ in range(df1.shape[0])]

        df2 = evaluate_single_plagiarism(df, s2, s1, False, context)
        df2["orig"] = [s2 for _ in range(df2.shape[0])]
        df2["target"] = [s1 for _ in range(df2.shape[0])]

        if merge:
            merged_df = pd.concat([merged_df, df1, df2])
        else:
            dfs_to_plot.append(pd.concat([df1, df2]))

    remap_dict = {'Differences relative ranking': 'Rel diff', 'Distances relative ranking': 'Rel dist'}

    sns.set_theme()
    sns.set_context(context)

    if merge:
        merged_df = (merged_df
                     >> dfply.gather("type", "value", ["Differences relative ranking", "Distance relative ranking"])
                     >> dfply.mutate(type=dfply.X['type'].apply(remap_dict.get))
                     )
        sns.displot(data=merged_df,
                    col="target",
                    row="orig",
                    x="value",
                    hue="type",
                    kind='hist',
                    stat='proportion',
                    col_order=merged_df.orig.unique(),
                    row_order=merged_df.orig.unique())

        plt.gca().yaxis.set_major_formatter(PercentFormatter(1))

        plt.savefig(os.path.join(data_path, "debug_outputs", "plagiarism_plot.png"))
        plt.show()
    else:
        for i, df in enumerate(dfs_to_plot):
            df = (df
                  >> dfply.gather("type", "value", ["Differences relative ranking", "Distance relative ranking"])
                  >> dfply.mutate(type=dfply.X['type'].apply(remap_dict.get))
                  )
            s1 = list(set(df["Style"]))[0]
            s2 = list(set(df["Style"]))[1]

            for styles_combination in [[s1, s2], [s2, s1]]:
                plagiarism_plot(df, styles_combination, context)
    return merged_df, dfs_to_plot


def evaluate_single_intervals_distribution(df, orig, dest, plot=True, context='talk'):
    distances_df = get_interval_distances_table(df, orig, dest)

    if plot:
        sns.set_theme()
        sns.set_context(context)
        sns.kdeplot(data=distances_df, x="log(tt/ot)")
        sns.displot(data=distances_df, x="log(ot/oo)", kind="kde")
        plt.title(f'Interval distribution of \n{orig} transformed to {dest}')
        plt.savefig(os.path.join(data_path, "debug_outputs", f"intervals_{orig}_to_{dest}.png"))
        plt.show()
    return distances_df


def get_intervals_results(df: pd.DataFrame, results: dict, orig: str, target: str, presentation_context='talk'):
    """
    Add to results how is the proportion of improvement (how many rolls got away from the old style and how many got
    closer to the new one).

    :param df: dataframe with distances values. It must have at least the columns 'orig', 'target', 'type' and 'value'.
    :param results: dictionary where to save the results.
    :param orig: name of a style to analyze.
    :param target: name of the other style to analyze.
    :param presentation_context: plot context ('talk' as default, 'paper' or 'poster').
    :return: It returns a DataFrame with columns 'Transference' and 'Improvement ratio'
    """
    for styles_combination in [[orig, target], [target, orig]]:
        intervals_plot(df, styles_combination, presentation_context)

        df_s1_to_s2 = df[(df['orig'] == styles_combination[0]) & (df['target'] == styles_combination[1])]
        df_get_away = df_s1_to_s2[df_s1_to_s2['type'] == "log(d(m's')/d(ms')) (< 0)\n Got away from the old style"]
        df_get_closer = df_s1_to_s2[df_s1_to_s2['type'] == "log(d(ms')/d(ms)) (> 0)\n Got closer to the new style"]
        results[f"{styles_combination[0]} to {styles_combination[1]} got away"] = \
            df_get_away[df_get_away['value'] < 0].shape[0] / df_get_away.shape[0]
        results[f"{styles_combination[0]} to {styles_combination[1]} got closer"] = \
            df_get_closer[df_get_closer['value'] > 0].shape[0] / df_get_closer.shape[0]
    return pd.DataFrame({"Transference": [k for k in results.keys()],
                         "Improvement ratio": [v for v in results.values()]})


def evaluate_multiple_intervals_distribution(dfs: List[pd.DataFrame], merge, context='talk'):
    """
    Estos dfs provendrían de cada df de ida y vuelta. Es decir, serían 6 dfs distintos.
    Considerando esto, en cada df voy a tener 2 estilos, así que evalúo single con ambos.
    """
    merged_df = pd.DataFrame()
    dfs_to_plot = []
    for df in dfs:
        s1 = list(set(df["Style"]))[0]
        s2 = list(set(df["Style"]))[1]

        df1 = evaluate_single_intervals_distribution(df, s1, s2, False, context)
        df1["orig"] = [s1 for _ in range(df1.shape[0])]
        df1["target"] = [s2 for _ in range(df1.shape[0])]

        df2 = evaluate_single_intervals_distribution(df, s2, s1, False, context)
        df2["orig"] = [s2 for _ in range(df2.shape[0])]
        df2["target"] = [s1 for _ in range(df2.shape[0])]

        if merge:
            merged_df = pd.concat([merged_df, df1, df2])
        else:
            dfs_to_plot.append(pd.concat([df1, df2]))

    remap_dict = {'log(tt/ot)': "log(d(m's')/d(ms')) (< 0)\n Got away from the old style",
                  'log(ot/oo)': "log(d(ms')/d(ms)) (> 0)\n Got closer to the new style"}

    results = {}
    df_results = pd.DataFrame

    if merge:
        merged_df = (merged_df
                     >> dfply.gather("type", "value", ["log(tt/ot)", "log(ot/oo)"])
                     >> dfply.mutate(type=dfply.X['type'].apply(remap_dict.get))
                     )

        intervals_plot(merged_df, merged_df['orig'].unique(), context)

    else:
        for i, df in enumerate(dfs_to_plot):
            df = (df
                  >> dfply.gather("type", "value", ["log(tt/ot)", "log(ot/oo)"])
                  >> dfply.mutate(type=dfply.X['type'].apply(remap_dict.get))
                  )

            s1 = list(set(df["Style"]))[0]
            s2 = list(set(df["Style"]))[1]

            table = get_intervals_results(df, results, s1, s2, context)
            df_results = table if df_results.empty else pd.concat([df_results, table])

    return merged_df, dfs_to_plot, df_results


if __name__ == "__main__":
    init(4)
    # df1 = load_pickle(os.path.join(data_path, "embeddings/brmf_4b/df_transferred_Bach_ragtime.pkl"))
    # df2 = load_pickle(os.path.join(data_path, "embeddings/brmf_4b/df_transferred_ragtime_Bach.pkl"))
    # df = pd.concat([df1, df2], axis=0)
    #
    # df = get_plagiarism_ranking_table(df)
    # save_pickle(df, f"{datasets_debug_path}/plagiarism_ranking_table")
    df = load_pickle(f"{datasets_debug_path}/plagiarism_ranking_table")
    df.to_csv(f"{data_path}/debug_outputs/plagiarism_ranking_table.csv")
