import os
from collections import Counter
from typing import List
from IPython.display import display


import dfply
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.ticker import PercentFormatter

from evaluation.metrics.intervals import get_interval_distances_table
from evaluation.metrics.plagiarism import get_most_similar_roll, get_plagiarism_ranking_table
from model.colab_tension_vae.params import init
from utils.audio_management import display_audio, save_audio, PlayMidi
from utils.files_utils import data_path, datasets_debug_path, load_pickle
from utils.plots_utils import intervals_plot, single_plagiarism_plot, plot_characteristics


def evaluate_model(dfs, plagiarism_args=None, intervals_args=None, eval_path=data_path):
    if not plagiarism_args is None:
        merge_pl, cache_path, context, by_distance, thold = False, None, 'talk', True, 1
        for k, v in plagiarism_args.items():
            if k == "merge": merge_pl = v
            elif k == "cache_path": cache_path = v
            elif k == "context": context = v
            elif k == "by_distance": by_distance = v
            elif k == "thold": thold = v
        print("===== Evaluating plagiarism =====")
        merged_df, table, p_successful_rolls = evaluate_multiple_plagiarism(dfs, merge_pl, cache_path, context, by_distance, thold)
        print(merged_df)
        print(table)

    if not intervals_args is None:
        merge_i, context = False, 'talk'
        for k, v in plagiarism_args.items():
            if k == "merge": merge_i = v
            elif k == "context": context = v
        print("===== Evaluate interval distributions =====")
        merged_df, df_to_plot, table, i_successful_rolls = evaluate_multiple_intervals_distribution(dfs, merge_i, context)
        print(merged_df)
        print(df_to_plot)
        print(table)
        # plot_characteristics(df_to_plot, )

    if not (plagiarism_args is None or intervals_args is None):
        successful_rolls = pd.merge(p_successful_rolls, i_successful_rolls, how="inner", on=["Style", "Title"])
        successful_rolls.dropna(inplace=True)
        for _, row in successful_rolls.iterrows():
            # display(PlayMidi(row["roll"].midi[:-3] + 'mid'))

            audio_file = save_audio(name=f"{row['Title']}_to_{row['target_x']}",
                                    pm=row["Transferred"].midi,
                                    path=eval_path)[:-3] + 'mid'

            # display_audio(eval_path + audio_file)


def calculate_resume_table(df, thold=1):
    """
    :param df: DataFrame of rolls with absolute positions in a plagiarism ranking.
    :param thold: last position in the ranking to consider as a 'winner'. If it is smaller than 1, the threshold will be
     the proportion of songs that can be better in ranking.
    :return: a DataFrame with columns: original "Style", "Target" style and "Percentage of winners".
    """
    winners = Counter()
    total = Counter()

    for _, r in df.iterrows():
        transformation = (r.Style, r.target)
        total[transformation] += 1

    for _, r in df.iterrows():
        transformation = (r.Style, r.target)
        if thold >= 1:
            winners[transformation] += r['value'] <= thold
        else:
            winners[transformation] += r['value'] <= thold * total[transformation]

    table = {"Style": [], "Target": [], "Percentage of winners": []}
    for transformation, w in winners.items():
        table["Style"].append(transformation[0])
        table["Target"].append(transformation[1])
        table["Percentage of winners"].append(w / total[transformation] * 100)

    return pd.DataFrame(table)


def evaluate_plagiarism_coincidences(df, direction, by_distance=False) -> float:
    rolls = list(df['rolls'])
    base_rolls = df[direction]
    titles = list(df['Title'])

    similarities = [title == get_most_similar_roll(base_roll, rolls, by_distance).song.name
                    for title, base_roll in zip(titles, base_rolls)]
    return sum(similarities) / len(similarities)


def evaluate_single_plagiarism(df, orig, dest, cache_path, by_distance=False, plot=True, context='talk'):
    plagiarism_df = get_plagiarism_ranking_table(df, cache_path=cache_path, by_distance=by_distance)

    if plot:
        kind = "Distance" if by_distance else "Differences"

        sns.set_theme()
        sns.set_context(context)
        sns.displot(data=plagiarism_df, x=f"{kind} position")
        # sns.displot(data=plagiarism_df, x="Distance position", kind="kde")
        plt.title(f'Plagiarism ranking of {orig} transformed to {dest}')
        plt.show()
    return plagiarism_df


def evaluate_multiple_plagiarism(dfs: List[pd.DataFrame], merge, cache_path, context='talk', by_distance=True, thold=1):
    """
    Estos dfs provendrían de cada df de ida y vuelta. Es decir, serían 6 dfs distintos.
    Considerando esto, en cada df voy a tener 2 estilos, así que evalúo single con ambos.
    """
    kind = "Distance" if by_distance else "Differences"
    merged_df = pd.DataFrame()
    dfs_with_rank = []

    for df in dfs:
        s1 = list(set(df["Style"]))[0]
        s2 = list(set(df["Style"]))[1]

        df1 = evaluate_single_plagiarism(df, s1, s2, cache_path, by_distance=by_distance, plot=False, context=context)
        df1["target"] = [s2 if s1 == df1["Style"][i] else s1 for i in range(df1.shape[0])]

        if merge:
            merged_df = pd.concat([merged_df, df1])  # , df2])
        else:
            dfs_with_rank.append(df1)  # pd.concat([df1, df2]))

    remap_dict = {f'{kind} relative ranking': f'Rel {kind[:4]}', f'{kind} position': f'Abs {kind[:4]}'}

    sns.set_theme()
    sns.set_context(context)

    table_results = pd.DataFrame()
    successful_rolls = pd.DataFrame()

    if merge:
        merged_df = (merged_df
                     >> dfply.gather("type", "value", [f"{kind} relative ranking"])
                     >> dfply.mutate(type=dfply.X['type'].apply(remap_dict.get))
                     )
        sns.displot(data=merged_df,
                    col="target",
                    row="Style",
                    x="value",
                    hue="type",
                    kind='hist',
                    stat='proportion',
                    col_order=merged_df.Style.unique(),
                    row_order=merged_df.Style.unique())

        plt.gca().yaxis.set_major_formatter(PercentFormatter(1))

        plt.savefig(os.path.join(data_path, "debug_outputs", "single_plagiarism_plot.png"))
        plt.show()
    else:
        for i, df in enumerate(dfs_with_rank):
            successful_rolls = pd.concat([successful_rolls, df[df[f"{kind} position"] == 1]])
            df_abs = (df
                      >> dfply.gather("type", "value", [f"{kind} position"])
                      >> dfply.mutate(type=dfply.X['type'].apply(remap_dict.get))
                      )

            table = calculate_resume_table(df_abs, thold)
            # table = get_plagiarism_results(df_abs, s1, s2, by_distance=by_distance, presentation_context=context)
            single_plagiarism_plot(df, context, by_distance)
            table_results = pd.concat([table_results, table])


    table_results.sort_values(by="Percentage of winners", ascending=False, inplace=True)
    return merged_df, table_results, successful_rolls


def evaluate_single_intervals_distribution(df, orig, dest, plot=True, context='talk'):
    distances_df = get_interval_distances_table(df, orig, dest)

    if plot:
        sns.set_theme()
        sns.set_context(context)
        sns.kdeplot(data=distances_df, x="log(m's/ms)")
        sns.displot(data=distances_df, x="log(m's'/ms')", kind="kde")
        plt.title(f'Interval distribution of \n{orig} transformed to {dest}')
        plt.savefig(os.path.join(data_path, "debug_outputs", f"intervals_{orig}_to_{dest}.png"))
        plt.show()
    return distances_df


def get_intervals_results(df: pd.DataFrame, orig: str, target: str, presentation_context='talk'):
    """
    Calculate the percentage of rolls that improve with the transformation (how many rolls got away from the old style
     and how many got closer to the new one).

    :param df: dataframe with distances values. It must have at least the columns 'orig', 'target', 'type' and 'value'.
    :param orig: name of a style to analyze.
    :param target: name of the other style to analyze.
    :param presentation_context: plot context ('talk' as default, 'paper' or 'poster').
    :return: It returns a DataFrame with columns 'Transference', '% got away' and '% got closer'
    """
    results = {}
    for styles_comb in [[orig, target], [target, orig]]:
        df_s1_to_s2 = df[(df['Style'] == styles_comb[0]) & (df['target'] == styles_comb[1])]

        df_got_away = df_s1_to_s2[df_s1_to_s2['type'] == "log(d(m',s)/d(m,s)) (> 0)\n Got away from the old style"]
        df_got_closer = df_s1_to_s2[df_s1_to_s2['type'] == "log(d(m',s')/d(m,s')) (< 0)\n Got closer to the new style"]
        intervals_plot(df_s1_to_s2, styles_comb, presentation_context)

        results[f"{styles_comb[0]} to {styles_comb[1]}"] = \
            df_got_away[df_got_away['value'] > 0].shape[0] / df_got_away.shape[0],\
            df_got_closer[df_got_closer['value'] < 0].shape[0] / df_got_closer.shape[0]

    return pd.DataFrame({"Transference": [k for k in results.keys()],
                         "% got away": [100 * v[0] for v in results.values()],
                         "% got closer": [100 * v[1] for v in results.values()],
                         })


def evaluate_multiple_intervals_distribution(dfs: List[pd.DataFrame], merge=False, context='talk'):
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
        df1["target"] = [s2 for _ in range(df1.shape[0])]

        df2 = evaluate_single_intervals_distribution(df, s2, s1, False, context)
        df2["target"] = [s1 for _ in range(df2.shape[0])]

        if merge:
            merged_df = pd.concat([merged_df, df1, df2])
        else:
            dfs_to_plot.append(pd.concat([df1, df2]))

    remap_dict = {"log(m's/ms)": "log(d(m',s)/d(m,s)) (> 0)\n Got away from the old style",
                  "log(m's'/ms')": "log(d(m',s')/d(m,s')) (< 0)\n Got closer to the new style"}

    table_results = pd.DataFrame()
    successful_rolls = pd.DataFrame()
    df_to_plot = pd.DataFrame()

    if merge:
        merged_df = (merged_df
                     >> dfply.gather("type", "value", ["log(m's/ms)", "log(m's'/ms')"])
                     >> dfply.mutate(type=dfply.X['type'].apply(remap_dict.get))
                     )

        intervals_plot(merged_df, merged_df['Style'].unique(), context)

    else:
        for i, df in enumerate(dfs_to_plot):
            successful_rolls = pd.concat([successful_rolls, df[df["log(m's'/ms')"] < 0]])
            df = (df
                  >> dfply.gather("type", "value", ["log(m's/ms)", "log(m's'/ms')"])
                  >> dfply.mutate(type=dfply.X['type'].apply(remap_dict.get))
                  )

            s1 = list(set(df["Style"]))[0]
            s2 = list(set(df["Style"]))[1]

            table = get_intervals_results(df, s1, s2, context)
            table_results = pd.concat([table_results, table])
            df_to_plot = pd.concat([df_to_plot, df])

    table_results.sort_values(by=["% got closer"], ascending=False, inplace=True)
    return merged_df, df_to_plot, table_results, successful_rolls


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
