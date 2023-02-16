import os
from collections import Counter
from typing import List

import dfply
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from data_analysis.statistics import rhythmic_closest_style, melodic_closest_style
from evaluation.metrics.musicality import information_rate
from evaluation.metrics.plagiarism import get_most_similar_roll
from utils.files_utils import data_path
from utils.plots_utils import bigrams_plot, plagiarism_plot, plot_IR_distributions, plot_fragments_distributions
from data_analysis.plots import plot_closeness


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


def evaluate_plagiarism(df: pd.DataFrame, orig, dest, eval_dir, by_distance=False, context='talk', thold=1):
    sns.set_theme()
    sns.set_context(context)
    kind = "Distance" if by_distance else "Differences"
    remap_dict = {f'{kind} relative ranking': f'Rel {kind[:4]}', f'{kind} position': f'Abs {kind[:4]}'}

    df["target"] = [dest if orig == df["Style"][i] else orig for i in range(df.shape[0])]

    sorted_df = df.sort_values(by=[f"{kind} position"])
    df_abs = (df
              >> dfply.gather("type", "value", [f"{kind} position"])
              >> dfply.mutate(type=dfply.X['type'].apply(remap_dict.get))
              )

    table_results = calculate_resume_table(df_abs, thold)
    # table = get_plagiarism_results(df_abs, s1, s2, by_distance=by_distance, presentation_context=context)
    plagiarism_plot(df, orig, dest, by_distance, eval_dir, context)

    table_results.sort_values(by="Percentage of winners", ascending=False, inplace=True)
    return table_results, sorted_df


def plot_intervals_improvements(orig, dest, interval_distances, plot_path, context='talk'):
    sns.set_theme()
    sns.set_context(context)
    sns.kdeplot(data=interval_distances, x="log(m's/ms)")
    sns.displot(data=interval_distances, x="log(m's'/ms')", kind="kde")
    plt.title(f'Interval distribution of \n{orig} transformed to {dest}')
    plt.savefig(os.path.join(data_path, plot_path, f"intervals_{orig}_to_{dest}.png"))
    # plt.show()


def get_bigrams_results(df: pd.DataFrame, orig: str, target: str, eval_dir: str, plot_name: str,
                        presentation_context='talk'):
    """
    Calculate the percentage of rolls that improve with the transformation (how many rolls got away from the old style
     and how many got closer to the new one).

    :param df: dataframe with distances values. It must have at least the columns 'orig', 'target', 'type' and 'value'.
    :param orig: name of a style to analyze.
    :param target: name of the other style to analyze.
    :param eval_dir: directory where save the plots
    :param plot_name: name of the plot (used for the title and the file name)
    :param presentation_context: plot context ('talk' as default, 'paper' or 'poster').
    :return: It returns a DataFrame with columns 'Transference', '% got away' and '% got closer'
    """
    results = {}
    df_s1_to_s2 = df[(df['Style'] == orig) & (df['target'] == target)]

    df_got_away = df_s1_to_s2[df_s1_to_s2['type'] == "log(d(m',s)/d(m,s)) (> 0)\n Got away from the old style"]
    df_got_closer = df_s1_to_s2[df_s1_to_s2['type'] == "log(d(m',s')/d(m,s')) (< 0)\n Got closer to the new style"]
    bigrams_plot(df_s1_to_s2, [orig, target], eval_dir, plot_name, presentation_context)

    results[f"{orig} to {target}"] = df_got_away[df_got_away['value'] > 0].shape[0] / df_got_away.shape[0], \
                                     df_got_closer[df_got_closer['value'] < 0].shape[0] / df_got_closer.shape[0]

    return pd.DataFrame({"Transference": [k for k in results.keys()],
                         "% got away": [100 * v[0] for v in results.values()],
                         "% got closer": [100 * v[1] for v in results.values()],
                         })


def evaluate_bigrams_distribution(bigram_distances, orig, dest, eval_path, plot_name, context='talk'):
    """
    Estos dfs provendrían de cada df de ida y vuelta. Es decir, serían 6 dfs distintos.
    Considerando esto, en cada df voy a tener 2 estilos, así que evalúo single con ambos.
    """
    plot_intervals_improvements(orig, dest, bigram_distances, eval_path, context)

    remap_dict = {"log(m's/ms)": "log(d(m',s)/d(m,s)) (> 0)\n Got away from the old style",
                  "log(m's'/ms')": "log(d(m',s')/d(m,s')) (< 0)\n Got closer to the new style"}

    sorted_df = bigram_distances.sort_values(by=["log(m's'/ms')"])
    df = (bigram_distances
          >> dfply.gather("type", "value", ["log(m's/ms)", "log(m's'/ms')"])
          >> dfply.mutate(type=dfply.X['type'].apply(remap_dict.get))
          )

    table_results = get_bigrams_results(df, orig, dest, eval_path, plot_name, context)

    table_results.sort_values(by=["% got closer"], ascending=False, inplace=True)
    return table_results, sorted_df


def evaluate_musicality(df, orig, dest, plot_dir, permutations=10, alpha=0.1):
    def roll_IRs_permutations(roll, n) -> List[int]:
        melody_changes = np.argwhere(roll.get_melody_changes() == 1)
        bass_changes = np.argwhere(roll.get_bass_changes() == 1)

        melody_changes = melody_changes.reshape(melody_changes.shape[0])
        bass_changes = bass_changes.reshape(bass_changes.shape[0])

        ps = [roll.permute(melody_changes, bass_changes) for _ in range(n)]
        return list(map(information_rate, ps))

    def distribution(irs: List[int]):
        return {"mean": np.mean(irs), "std": np.std(irs)}

    df["IRs perm"] = df.apply(lambda row: roll_IRs_permutations(row['roll'], permutations), axis=1)
    # df["Distribution of probability of IRs"] = df.apply(lambda row: distribution(row['IRs of permutations']), axis=1)
    df["IRs mean"] = df.apply(lambda row: np.mean(row['IRs perm']), axis=1)
    df["IRs std"] = df.apply(lambda row: np.std(row['IRs perm']), axis=1)

    df["Distance difference"] = df.apply(lambda row:
                                         abs(row["IR trans"] - row["IRs mean"]) - abs(row["IR trans"] - row["IR orig"]),
                                         axis=1)

    plot_IR_distributions(df[["Style", "IR orig", "IR trans", "IRs perm"]], orig, dest, plot_dir)

    def hypothesis_test(df, alpha) -> pd.DataFrame:
        df["Hypothesis Test"] = df.apply(lambda row: row["IR orig"] < row["IRs mean"] - row["IRs std"]
                                                     or row["IR orig"] > row["IRs mean"] + row["IRs std"], axis=1)

        return df[df["Hypothesis Test"]]

    df_verified_hypothesis = hypothesis_test(df, alpha)
    sorted_df = df.sort_values(by=["Distance difference"])
    table_results = pd.DataFrame({
        "% IR from original roll has low probability to be in the IRs permutations distribution": [
            df_verified_hypothesis.shape[0] / df.shape[0] * 100],
        "% have more musicality than permutations": [sorted_df.shape[0] / df.shape[0] * 100]
    })

    return table_results, sorted_df



def evaluate_style_belonging(rhythmic_bigram_distances, melodic_bigram_distances, styles, orig, dest, eval_path, context='talk'):
    """
    Calculate for each transformed roll to which style it sames to belong by comparing its distance with the
    characteristic entropy matrix of rhythmic and melodic bigrams. Distance can be linear distance or Kullback-Leibler
    divergence.
    """

    rhythmic_bigram_distances["Rhythmic closest style (linear)"] = rhythmic_bigram_distances.apply(lambda row: rhythmic_closest_style(row["m'"], styles), axis=1)
    rhythmic_bigram_distances["Rhythmic closest style (kl)"] = rhythmic_bigram_distances.apply(lambda row: rhythmic_closest_style(row["m'"], styles, method='kl'), axis=1)
    rhythmic_bigram_distances["Rhythmic closest style (probability)"] = rhythmic_bigram_distances.apply(lambda row: rhythmic_closest_style(row["m'"], styles, method='probability'), axis=1)
    melodic_bigram_distances["Melodic closest style (linear)"] = melodic_bigram_distances.apply(lambda row: melodic_closest_style(row["m'"], styles), axis=1)
    melodic_bigram_distances["Melodic closest style (kl)"] = melodic_bigram_distances.apply(lambda row: melodic_closest_style(row["m'"], styles, method='kl'), axis=1)
    melodic_bigram_distances["Melodic closest style (probability)"] = melodic_bigram_distances.apply(lambda row: melodic_closest_style(row["m'"], styles, method='probability'), axis=1)

    plot_closeness(rhythmic_bigram_distances, melodic_bigram_distances, orig, dest, eval_path, context)


def evaluate_model(df, metrics, styles_char, eval_path=data_path, **kwargs):
    merge_pl, cache_path, context, by_distance, thold = False, None, 'talk', False, 1
    for k, v in kwargs.items():
        if k == "context":
            context = v
        elif k == "by_distance":
            by_distance = v
        elif k == "thold":
            thold = v

    orig, target = metrics["original_style"], metrics["target_style"]

    print("===== Evaluating Style belonging =====")
    evaluate_style_belonging(metrics["rhythmic_bigrams"],
                             metrics["intervals"],
                             styles_char,
                             orig, target, eval_path, context)


    print("===== Evaluating rhythmic bigrams distributions =====")
    r_table, r_sorted_df = evaluate_bigrams_distribution(metrics["rhythmic_bigrams"],
                                                         orig, target, eval_path + '/rhythmic', f"Rhythmic bigrams {orig} to {target}", context)
    r_sorted_df["RhythmicStyle rank"] = range(r_sorted_df.shape[0])
    print(r_table)

    print("===== Evaluating interval distributions =====")
    i_table, i_sorted_df = evaluate_bigrams_distribution(metrics["intervals"],
                                                         orig, target, eval_path + '/melodic', f"Interval {orig} to {target}", context)
    i_sorted_df["IntervalStyle rank"] = range(i_sorted_df.shape[0])
    print(i_table)

    plot_fragments_distributions(df, styles_char, eval_path, f"Transformation_distribution_{orig}_to_{target}")

    print("===== Evaluating plagiarism =====")
    p_table, p_sorted_df = evaluate_plagiarism(metrics["plagiarism"], orig, target, eval_path, by_distance, context, thold)
    p_sorted_df["Plagiarism rank"] = range(p_sorted_df.shape[0])
    print(p_table)

    print("===== Evaluating musicality =====")
    ir_table, ir_sorted_df = evaluate_musicality(metrics["musicality"], orig, target, eval_path)
    ir_sorted_df["IR rank"] = range(ir_sorted_df.shape[0])
    print(ir_table)

    # print("===== Selecting audios of successful rolls =====")
    # successful_rolls = pd.merge(p_sorted_df, i_sorted_df, how="inner",
    #                             on=["Style", "Title", "roll", "NewRoll", "target"])
    # if successful_rolls.shape[0] != 0:
    #     successful_rolls = pd.merge(successful_rolls, r_sorted_df, how="inner",
    #                                 on=["Style", "Title", "roll", "NewRoll", "target"])
    # if successful_rolls.shape[0] != 0:
    #     successful_rolls = pd.merge(successful_rolls, ir_sorted_df, how="inner",
    #                                 on=["Style", "Title", "roll", "NewRoll"])
    #
    # successful_rolls = successful_rolls[["Style", "Title", "roll", "NewRoll", "target",
    #                                      "IntervalStyle rank", "RhythmicStyle rank", "IR rank", "Plagiarism rank"]]
    #
    # successful_rolls["Merged rank"] = sum([successful_rolls["IntervalStyle rank"],
    #                                        successful_rolls["RhythmicStyle rank"],
    #                                        successful_rolls["IR rank"],
    #                                        successful_rolls["Plagiarism rank"]])
    # successful_rolls.sort_values(by=["Merged rank"], inplace=True)

    return {#"Merged": successful_rolls,
            "IntervalStyle": i_sorted_df,
            "RhythmicStyle": r_sorted_df,
            "IR": ir_sorted_df,
            "Plagiarism": p_sorted_df}, \
        {"interval": i_table,
         "rhythmic": r_table,
         "IR": ir_table,
         "plagiarism": p_table}
    # return  {}, {}
