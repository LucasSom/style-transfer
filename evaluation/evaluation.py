from collections import Counter
from typing import List, Dict, Tuple

import dfply
import numpy as np
import pandas as pd

from data_analysis.assemble_data import optimal_transport, \
    linear_distance, kl, belonging_probability, joined_closest_style
from evaluation.metrics.intervals import matrix_of_adjacent_intervals
from evaluation.metrics.musicality import information_rate
from evaluation.metrics.plagiarism import get_most_similar_roll
from evaluation.metrics.rhythmic_bigrams import matrix_of_adjacent_rhythmic_bigrams
from roll.guoroll import roll_permutations
from utils.files_utils import data_path
from utils.plots_utils import bigrams_plot, plagiarism_plot, plot_IR_distributions, plot_intervals_improvements
from data_analysis.dataset_plots import plot_closeness, plot_musicality_distribution, plot_distances


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
    kind = "Distance" if by_distance else "Differences"
    remap_dict = {f'{kind} relative ranking': f'Rel {kind[:4]}', f'{kind} position': f'Abs {kind[:4]}'}

    df["target"] = [dest if orig == df["Style"][i] else orig for i in range(df.shape[0])]

    sorted_df = df.sort_values(by=[f"{kind} position"])
    sorted_df.rename({f'{kind} position': "Plagiarism rank"}, inplace=True)
    avg_plagiarism_rank = sorted_df.loc[:, f"{kind} relative ranking"].mean()

    df_abs = (df
              >> dfply.gather("type", "value", [f"{kind} position"])
              >> dfply.mutate(type=dfply.X['type'].apply(remap_dict.get))
              )

    table_results = calculate_resume_table(df_abs, thold)
    # table = get_plagiarism_results(df_abs, s1, s2, by_distance=by_distance, presentation_context=context)
    plagiarism_plot(df, orig, dest, by_distance, eval_dir, context)

    table_results.sort_values(by="Percentage of winners", ascending=False, inplace=True)
    return table_results, sorted_df, avg_plagiarism_rank


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


def evaluate_IR(df, orig, dest, plot_dir, permutations=10, alpha=0.1):
    def roll_IRs_permutations(roll, n) -> List[int]:
        ps = roll_permutations(roll, n)
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


def count_musicality(df_test, df_permutations):
    """
    It calculates the proportion of generated rolls that are more musicals than i permutations of the original roll
    """
    def perm_less_musical(row):
        t, roll_id = row['Title'], row['roll_id']
        ps = df_permutations[(df_permutations['Title'] == t) & (df_permutations['roll_id'] == roll_id)]
        better_than_ps = [abs(d) > abs(row['Joined musicality difference (probability)'])
                          for d in ps['Joined musicality difference (probability)']]
        return sum(better_than_ps) / ps.shape[0] * 100

    df_test["% permutations that are less musical"] = df_test.apply(perm_less_musical, axis=1)

    return df_test


def evaluate_musicality(df_train, df_test, melodic_distribution, rhythmic_distribution, eval_dir, plot_suffix='',
                        only_probability=False, only_joined=True, n_permutations=5):
    if only_probability:
        methods = [('probability', belonging_probability)]
    else:
        methods = [('linear', linear_distance), ('kl', kl), ('ot', optimal_transport), ('probability', belonging_probability)]

    df_train["Melodic bigram matrix"] = df_train.apply(lambda r: matrix_of_adjacent_intervals(r["roll"])[0], axis=1)
    df_train["Rhythmic bigram matrix"] = df_train.apply(lambda r: matrix_of_adjacent_rhythmic_bigrams(r["roll"])[0], axis=1)

    # Armo las permutaciones
    df_permutations = {'Title': [], 'roll_id': [], 'roll': [], "Melodic bigram matrix": [], "Rhythmic bigram matrix": []}
    for _, row in df_train.iterrows():
        ps = roll_permutations(row['roll'], n=n_permutations)
        df_permutations['Title'] += n_permutations * [row['Title']]
        df_permutations['roll_id'] += n_permutations * [row['roll_id']]
        df_permutations['roll'] += ps
        df_permutations['Melodic bigram matrix'] += [matrix_of_adjacent_intervals(p)[0] for p in ps]
        df_permutations['Rhythmic bigram matrix'] += [matrix_of_adjacent_rhythmic_bigrams(p)[0] for p in ps]
    df_permutations = pd.DataFrame(df_permutations)

    # Calculo las diferencias (con cada método) con la matriz de musicalidad para cada df (df, train y permutaciones)
    for df in [df_train, df_test, df_permutations]:
        if 'Title' in df.columns and not 'roll_id' in df.columns:
            df = df.drop_duplicates(subset=['Title'])

        for method_name, func in methods:
            df[f'Melodic musicality difference ({method_name})'] = \
                df.apply(lambda r: func(melodic_distribution, r["Melodic bigram matrix"], True), axis=1)
            df[f'Rhythmic musicality difference ({method_name})'] = \
                df.apply(lambda r: func(rhythmic_distribution, r["Rhythmic bigram matrix"], False), axis=1)

            df[f'Joined musicality difference ({method_name})'] = \
                df.apply(lambda r:
                         r[f'Melodic musicality difference ({method_name})']
                         + r[f'Rhythmic musicality difference ({method_name})'], axis=1)

    # Ploteo la distribución de las diferencias de los df con la matriz de musicalidad
    plot_musicality_distribution({'train': df_train, 'test': df_test, 'permutations': df_permutations}, eval_dir,
                                 plot_suffix, only_probability=only_probability, only_joined=only_joined)

    df_test = count_musicality(df_test, df_permutations)
    avg_musicality_rank = df_test.loc[:, "% permutations that are less musical"].mean()
    table = {'% rolls that are more musical than all permutations': [df_test[df_test["% permutations that are less musical"] == 100]]}

    df_test['Joined musicality difference (probability)'] = df_test.apply(lambda row: abs(row['Joined musicality difference (probability)']), axis=1)
    sorted_df = df_test.sort_values(by=['Joined musicality difference (probability)'], ascending=True)
    sorted_df.rename({'Joined musicality difference (probability)': "Musicality rank"}, inplace=True, axis=1)
    return pd.DataFrame(table), sorted_df, avg_musicality_rank


def count_closest(df, style):
    df_style = df[df["target"] == style]
    return df_style[df_style["Joined closest style (ot)"] == style].shape[0] / df_style.shape[0] * 100


def styles_approach(df, styles, orig: str, dest: str) -> Tuple[Dict[str, int], Dict[str, List[float]]]:
    """
    It calculates how many rolls (in percentage) got closer to each style (w/o the original style).

    :param df: DataFrame with columns 'Style' 'm_orig_rhythmic' and 'm_orig_melodic'
    :param styles: dictionary that maps style names with Style objects
    :param orig: name of the original style
    :param dest: name of the destination style
    :return: a dictionary that maps styles with the percentages of rolls that are closer to the style.
    """
    improvements = {}
    distances = {}
    df = df[df["Style"] == orig]
    for name, style_obj in styles.items():
        if name != orig:
            df[f"Rhythmic closeness to {name} (orig)"] = df.apply(lambda row: optimal_transport(style_obj.rhythmic_bigrams_distribution, row["m_orig_rhythmic"], melodic=False), axis=1)
            df[f"Melodic closeness to {name} (orig)"] = df.apply(lambda row: optimal_transport(style_obj.intervals_distribution, row["m_orig_melodic"], melodic=True), axis=1)
            df[f"Joined closeness to {name} (orig)"] = df[f"Rhythmic closeness to {name} (orig)"] + df[f"Melodic closeness to {name} (orig)"]

            df[f"Rhythmic closeness to {name} (trans)"] = df.apply(lambda row: optimal_transport(style_obj.rhythmic_bigrams_distribution, row["m_trans_rhythmic"], melodic=False), axis=1)
            df[f"Melodic closeness to {name} (trans)"] = df.apply(lambda row: optimal_transport(style_obj.intervals_distribution, row["m_trans_melodic"], melodic=True), axis=1)
            df[f"Joined closeness to {name} (trans)"] = df[f"Rhythmic closeness to {name} (trans)"] + df[f"Melodic closeness to {name} (trans)"]

            df[f"Improvement of joined closeness to {name}"] = df[f"Joined closeness to {name} (trans)"] < df[f"Joined closeness to {name} (orig)"]
            df[f"Normalized distance to {name}"] = (df[f"Joined closeness to {name} (trans)"] - df[f"Joined closeness to {name} (orig)"]) / df[f"Joined closeness to {name} (orig)"]

            sub_df = df[df["target"] == dest]
            n = sub_df[sub_df[f"Improvement of joined closeness to {name}"]].shape[0]
            improvements[name] = n / sub_df.shape[0] * 100
            distances[name] = list(df[f"Normalized distance to {name}"])

    return improvements, distances
    # sns.heatmap(d, annot=True, fmt='d')
    # save_plot(plot_path, f"confusion_matrix_{orig}")


def evaluate_style_belonging(rhythmic_bigram_distances, melodic_bigram_distances, styles, orig, dest, eval_path, context='talk'):
    """
    Calculate for each transformed roll to which style it sames to belong by comparing its optimal transport distance
    with the characteristic entropy matrix of rhythmic and melodic bigrams.
    """
    common_columns = ['Style', 'Title', 'roll', 'Reconstruction', 'NewRoll', 'target']
    joined_df = rhythmic_bigram_distances[common_columns + ["m'", "m"]].merge(melodic_bigram_distances[common_columns + ["m'", "m"]],
                                                                        on=common_columns, how='inner')
    joined_df.rename(columns={"m'_x": 'm_trans_rhythmic', "m'_y": 'm_trans_melodic', "m_x": "m_orig_rhythmic", "m_y": "m_orig_melodic"},
                     inplace=True)

    styles_approach_dict, distances_dict = styles_approach(joined_df, styles, orig, dest)

    joined_df = joined_df[joined_df["target"] == dest]
    joined_df["Joined closest style (ot)"] = joined_df.apply(lambda row: joined_closest_style(row["m_trans_melodic"], row["m_trans_rhythmic"], styles, method='ot'), axis=1)
    joined_df["Style rank"] = joined_df.apply(lambda row:
                                              abs(optimal_transport(styles[dest].intervals_distribution, row["m_trans_melodic"], True))
                                            + abs(optimal_transport(styles[dest].rhythmic_bigrams_distribution, row["m_trans_rhythmic"], False)),
                                              axis=1)
    joined_df.sort_values(by=['Style rank'], ascending=True)

    # plot_closeness(rhythmic_bigram_distances, melodic_bigram_distances, orig, dest, eval_path, context)
    plot_closeness(joined_df, orig, dest, eval_path, context, only_joined_ot=True)
    plot_distances(distances_dict, orig, dest, eval_path, context)

    table = {f'% rolls whose closest style is the target ({dest})': [count_closest(joined_df, dest)]}
    return pd.DataFrame(table), joined_df, styles_approach_dict


def evaluate_model(df, metrics, styles_char, melodic_musicality_distribution, rhythmic_musicality_distribution,
                   eval_path=data_path, **kwargs):
    merge_pl, cache_path, context, thold = False, None, 'talk', 1
    for k, v in kwargs.items():
        if k == "context":
            context = v
        elif k == "thold":
            thold = v

    orig, target = metrics["original_style"], metrics["target_style"]

    print("===== Evaluating Style belonging =====")
    s_table, s_df, styles_approach_dict = evaluate_style_belonging(metrics["rhythmic_bigrams"],
                                                                   metrics["intervals"],
                                                                   styles_char,
                                                                   orig, target, eval_path, context)

    print("===== Evaluating plagiarism (dist) =====")
    by_distance = True
    p_dist_table, p_dist_sorted_df, avg_plagiarism_dist_rank = evaluate_plagiarism(metrics["plagiarism-dist"], orig, target, eval_path, by_distance, context, thold)
    p_dist_sorted_df["Plagiarism (dist) rank"] = range(p_dist_sorted_df.shape[0])

    print("===== Evaluating plagiarism (diff) =====")
    by_distance = False
    p_diff_table, p_diff_sorted_df, avg_plagiarism_diff_rank = evaluate_plagiarism(metrics["plagiarism-diff"], orig, target, eval_path, by_distance, context, thold)
    p_diff_sorted_df["Plagiarism (diff) rank"] = range(p_diff_sorted_df.shape[0])

    print("===== Evaluating musicality =====")
    df_test = df[["Style", "Title", "roll", "NewRoll", "roll_id", "Reconstruction"]]
    common_columns = ['Style', 'Title', 'roll', 'Reconstruction', 'NewRoll', 'target']
    joined_df = metrics["rhythmic_bigrams"][common_columns + ["m'"]].merge(metrics["intervals"][common_columns + ["m'"]], on=common_columns, how='inner')
    df_test = df_test.merge(joined_df[["Style", "Title", "m'_x", "m'_y"]], on=["Style", "Title"])
    df_test.rename(columns={"m'_x": 'Rhythmic bigram matrix', "m'_y": 'Melodic bigram matrix'}, inplace=True)

    mus_table, mus_sorted_df, avg_musicality_rank = evaluate_musicality(df, df_test, melodic_musicality_distribution,
                                                                        rhythmic_musicality_distribution, eval_path,
                                                                        f'_{orig}_to_{target}', only_probability=True)

    return {"Style": s_df,
            "Musicality": mus_sorted_df,
            "Plagiarism (dist)": p_dist_sorted_df,
            "Plagiarism (diff)": p_diff_sorted_df}, \
        {"Style": s_table,
         "Musicality": mus_table,
         "Plagiarism (dist)": p_dist_table,
         "Plagiarism (diff)": p_diff_table}, \
        {"Style": styles_approach_dict,
         "Musicality": avg_musicality_rank,
         "Plagiarism (dist)": avg_plagiarism_dist_rank,
         "Plagiarism (diff)": avg_plagiarism_diff_rank,
         'orig': orig, 'target': target}
