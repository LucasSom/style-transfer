import os
from collections import Counter
from typing import List, Union

import dfply
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from evaluation.metrics.plagiarism import get_most_similar_roll
from model.colab_tension_vae.params import init
from utils.audio_management import save_audio, display_audio
from utils.files_utils import data_path, datasets_debug_path, load_pickle
from utils.plots_utils import intervals_plot, plagiarism_plot, calculate_TSNEs, plot_tsnes_comparison, plot_tsne, \
    plot_embeddings


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


def evaluate_plagiarism(df: pd.DataFrame, orig, dest, context='talk', by_distance=False, thold=1):
    """
    Estos dfs provendrían de cada df de ida y vuelta. Es decir, serían 6 dfs distintos.
    Considerando esto, en cada df voy a tener 2 estilos, así que evalúo single con ambos.
    """
    sns.set_theme()
    sns.set_context(context)
    kind = "Distance" if by_distance else "Differences"
    remap_dict = {f'{kind} relative ranking': f'Rel {kind[:4]}', f'{kind} position': f'Abs {kind[:4]}'}

    df["target"] = [dest if orig == df["Style"][i] else orig for i in range(df.shape[0])]

    successful_rolls = df[df[f"{kind} position"] == 1]
    df_abs = (df
              >> dfply.gather("type", "value", [f"{kind} position"])
              >> dfply.mutate(type=dfply.X['type'].apply(remap_dict.get))
              )

    table_results = calculate_resume_table(df_abs, thold)
    # table = get_plagiarism_results(df_abs, s1, s2, by_distance=by_distance, presentation_context=context)
    plagiarism_plot(df, context, orig, dest, by_distance)

    table_results.sort_values(by="Percentage of winners", ascending=False, inplace=True)
    return table_results, successful_rolls


def plot_intervals_distribution(orig, dest, interval_distances, context='talk'):
    sns.set_theme()
    sns.set_context(context)
    sns.kdeplot(data=interval_distances, x="log(m's/ms)")
    sns.displot(data=interval_distances, x="log(m's'/ms')", kind="kde")
    plt.title(f'Interval distribution of \n{orig}? transformed to {dest}?')
    plt.savefig(os.path.join(data_path, "debug_outputs", f"intervals_{orig}_to_{dest}.png"))
    plt.show()


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
    df_s1_to_s2 = df[(df['Style'] == orig) & (df['target'] == target)]

    df_got_away = df_s1_to_s2[df_s1_to_s2['type'] == "log(d(m',s)/d(m,s)) (> 0)\n Got away from the old style"]
    df_got_closer = df_s1_to_s2[df_s1_to_s2['type'] == "log(d(m',s')/d(m,s')) (< 0)\n Got closer to the new style"]
    intervals_plot(df_s1_to_s2, [orig, target], presentation_context)

    results[f"{orig} to {target}"] = df_got_away[df_got_away['value'] > 0].shape[0] / df_got_away.shape[0], \
                                     df_got_closer[df_got_closer['value'] < 0].shape[0] / df_got_closer.shape[0]

    return pd.DataFrame({"Transference": [k for k in results.keys()],
                         "% got away": [100 * v[0] for v in results.values()],
                         "% got closer": [100 * v[1] for v in results.values()],
                         })


def evaluate_bigrams_distribution(interval_distances, orig, dest, context='talk'):
    """
    Estos dfs provendrían de cada df de ida y vuelta. Es decir, serían 6 dfs distintos.
    Considerando esto, en cada df voy a tener 2 estilos, así que evalúo single con ambos.
    """
    plot_intervals_distribution(orig, dest, interval_distances, context)

    remap_dict = {"log(m's/ms)": "log(d(m',s)/d(m,s)) (> 0)\n Got away from the old style",
                  "log(m's'/ms')": "log(d(m',s')/d(m,s')) (< 0)\n Got closer to the new style"}

    successful_rolls = interval_distances[interval_distances["log(m's'/ms')"] < 0]
    df_to_plot = (interval_distances
          >> dfply.gather("type", "value", ["log(m's/ms)", "log(m's'/ms')"])
          >> dfply.mutate(type=dfply.X['type'].apply(remap_dict.get))
          )

    table_results = get_intervals_results(df_to_plot, orig, dest, context)

    table_results.sort_values(by=["% got closer"], ascending=False, inplace=True)
    return df_to_plot, table_results, successful_rolls


def evaluate_rhythmic_bigrams(df: pd.DataFrame, orig, dest, plots_path, context='talk'):
    tsne_emb = calculate_TSNEs(df, column_discriminator="Style")[0]

    plot_tsnes_comparison(df, tsne_emb, plots_path)
    plot_tsne(df, tsne_emb, plots_path)


def evaluate_model(metrics, styles_char, eval_path=data_path, **kwargs):
    merge_pl, cache_path, context, by_distance, thold = False, None, 'talk', False, 1
    for k, v in kwargs.items():
        if k == "context": context = v
        elif k == "by_distance": by_distance = v
        elif k == "thold": thold = v


    print("===== Evaluating interval distributions =====")
    df_to_plot, table, i_successful_rolls = evaluate_bigrams_distribution(metrics["intervals"],
                                                                          metrics["original_style"],
                                                                          metrics["target_style"], context)
    print(table)
    plot_embeddings(df_to_plot, "m", {n: s.intervals_distribution for n, s in styles_char.items()},
                         f"{eval_path}/plots/intervals/m")
    plot_embeddings(df_to_plot, "m'", {n: s.intervals_distribution for n, s in styles_char.items()},
                         f"{eval_path}/plots/intervals/mt")

    print("===== Evaluating rhythmic bigrams distributions =====")
    df_to_plot, table, r_successful_rolls = evaluate_bigrams_distribution(metrics["rhythmic_bigrams"],
                                                                          metrics["original_style"],
                                                                          metrics["target_style"], context)
    print(table)
    plot_embeddings(df_to_plot, "m", {n: s.rhythmic_bigrams_distribution for n, s in styles_char.items()},
                         f"{eval_path}/plots/r_bigrams/m")
    plot_embeddings(df_to_plot, "m'", {n: s.rhythmic_bigrams_distribution for n, s in styles_char.items()},
                         f"{eval_path}/plots/r_bigrams/mt")

    print("===== Evaluating plagiarism =====")
    table, p_successful_rolls = evaluate_plagiarism(metrics["plagiarism"],
                                                    metrics["original_style"],
                                                    metrics["target_style"], context, by_distance, thold)
    print(table)

    print("===== Creating audios of succesfull rolls =====")
    successful_rolls = pd.merge(p_successful_rolls, i_successful_rolls, how="inner", on=["Style", "Title"])
    successful_rolls = pd.merge(successful_rolls, r_successful_rolls, how="inner", on=["Style", "Title"])
    successful_rolls.dropna(inplace=True)

    for _, row in successful_rolls.iterrows():
        # display(PlayMidi(row["roll"].midi[:-3] + 'mid'))
        audio_file = save_audio(name=f"{row['Title']}_to_{row['target_x']}",
                                pm=row["NewRoll"].midi,
                                path=eval_path)[:-3] + 'mid'
        display_audio(audio_file)


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
