import os
from typing import List

import dfply
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

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


def evaluate_multiple_plagiarism(dfs: List[pd.DataFrame], context='talk'):
    """
    Estos dfs provendrían de cada df de ida y vuelta. Es decir, serían 6 dfs distintos.
    Considerando esto, en cada df voy a tener 2 estilos, así que evalúo single con ambos.
    """
    merged_df = pd.DataFrame()
    for df in dfs:
        s1 = list(set(df["Style"]))[0]
        s2 = list(set(df["Style"]))[1]

        df1 = evaluate_single_plagiarism(df, s1, s2, False)
        df1["orig"] = [s1 for _ in range(df1.shape[0])]
        df1["target"] = [s2 for _ in range(df1.shape[0])]

        df2 = evaluate_single_plagiarism(df, s2, s1, False)
        df2["orig"] = [s2 for _ in range(df2.shape[0])]
        df2["target"] = [s1 for _ in range(df2.shape[0])]

        merged_df = pd.concat([merged_df, df1, df2])

    remap_dict = {'Differences relative ranking': 'Rel diff', 'Distances relative ranking': 'Rel dist'}
    merged_df = (merged_df
                 >> dfply.gather("type", "value", ["Differences relative ranking", "Distance relative ranking"])
                 >> dfply.mutate(type=dfply.X['type'].apply(remap_dict.get))
                 )

    sns.set_theme()
    sns.set_context(context)
    sns.histplot(data=merged_df,
                 col="target",
                 row="orig",
                 x="value",
                 hue="type",
                 col_order=merged_df.orig.unique(),
                 row_order=merged_df.orig.unique())
    plt.savefig(os.path.join(data_path, "debug_outputs", "plagiarism_plot.png"))
    plt.show()
    return merged_df


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

    remap_dict = {'log(tt/ot)': "log(d(m's')/d(ms')) (< 0)\n Get away from the old style",
                  'log(ot/oo)': "log(d(ms')/d(ms)) (> 0)\n Get closer to the new style"}

    sns.set_theme()
    sns.set_context(context)

    if merge:
        merged_df = (merged_df
                     >> dfply.gather("type", "value", ["log(tt/ot)", "log(ot/oo)"])
                     >> dfply.mutate(type=dfply.X['type'].apply(remap_dict.get))
                     )

        intervals_plot(merged_df, merged_df['orig'].unique(), context)
        return merged_df

    else:
        for i, df in enumerate(dfs_to_plot):
            df = (df
                  >> dfply.gather("type", "value", ["log(tt/ot)", "log(ot/oo)"])
                  >> dfply.mutate(type=dfply.X['type'].apply(remap_dict.get))
                  )

            s1 = list(set(df["Style"]))[0]
            s2 = list(set(df["Style"]))[1]

            for styles_combination in [[s1, s2], [s2, s1]]:
                intervals_plot(df, styles_combination, context)
        return dfs_to_plot


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
