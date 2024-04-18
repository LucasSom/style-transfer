import glob
import os
from typing import List

import numpy as np
import pandas as pd
import seaborn as sns
from numpy import float64

from utils.files_utils import load_pickle
from utils.plots_utils import save_plot


# def heatmap_style_evaluation(df, orig, plot_dir, context='talk'):
#     # sns.set_theme(context)
#     sns.heatmap(df, annot=True, fmt='g', vmin=0, vmax=100)
#     save_plot(plot_dir, f"heatmap_style_{orig}", f"Avg of % of approaches to styles from {orig}")


def overall_single_evaluation(metrics, plot_dir, kind='boxplot', context='notebook'):
    """
    :metrics: dict with keys "Style", "Musicality" and "Plagiarism" with their respective DataFrames to plot the heatmaps
    """
    sns.set_theme(context)

    # Plot heatmap for Style evaluation
    # for orig, df in metrics["Style"].items():
    #     heatmap_style_evaluation(df, orig, plot_dir, context)

    # Plot heatmap style closeness
    if kind == 'heatmap':
        sns.heatmap(metrics["Style"], annot=True, fmt=".2f", vmin=0, vmax=100)
        save_plot(plot_dir, f"heatmap_style", f"Promedios de % de acercamiento\na los nuevos estilos")
    elif kind == 'boxplot':
        sns.boxplot(metrics["Style"], orient="v")
        save_plot(plot_dir, "boxplot_style", "Distributions of style approaches")
    d = {s: [sum(metrics["Style"][s]) / 3] for s in metrics["Style"].columns}
    df_style = pd.DataFrame(d)
    df_style.to_csv(plot_dir + "/overall_style.csv", index=False)

    # Plot heatmap musicality
    if kind == 'heatmap':
        sns.heatmap(metrics["Musicality"], annot=True, fmt=".2f", vmin=0, vmax=100)
        save_plot(plot_dir, "heatmap_musicality",
                  "Promedios de % de permutaciones\nque son menos musicales que el generado")
    elif kind == 'boxplot':
        sns.boxplot(metrics["Musicality"], orient="v")
        save_plot(plot_dir, "boxplot_style", "Distributions of musicality")
    d = {s: [sum(metrics["Musicality"][s]) / 3] for s in metrics["Musicality"].columns}
    df_mus = pd.DataFrame(d)
    df_mus.to_csv(plot_dir + "/overall_musicality.csv", index=False)

    # Plot heatmap plagiarism (dist)
    if kind == 'heatmap':
        sns.heatmap(metrics["Plagiarism-dist"], annot=True, fmt=".2f", vmin=0, vmax=1)
        save_plot(plot_dir, "heatmap_plagiarism_dist",
                  "Promedios de % de rolls\nque son más diferentes que el generado\n(distancia)")
    elif kind == 'boxplot':
        sns.boxplot(metrics["Style"], orient="v")
        save_plot(plot_dir, "boxplot_style", "Distributions of plagiarism (distance)")
    d = {s: [sum(metrics["Plagiarism-dist"][s]) / 3] for s in metrics["Plagiarism-dist"].columns}
    df_plagiarism_dist = pd.DataFrame(d)
    df_plagiarism_dist.to_csv(plot_dir + "/overall_plagiarism_dist.csv", index=False)

    # Plot heatmap plagiarism (diff)
    if kind == 'heatmap':
        sns.heatmap(metrics["Plagiarism-diff"], annot=True, fmt=".2f", vmin=0, vmax=1)
        save_plot(plot_dir, "heatmap_plagiarism_diff",
                  "Promedios de % de rolls\nque son más diferentes que el generado\n(diferencia)")
    elif kind == 'boxplot':
        sns.boxplot(metrics["Style"], orient="v")
        save_plot(plot_dir, "boxplot_style", "Distributions of plagiarism (difference)")
    d = {s: [sum(metrics["Plagiarism-diff"][s]) / 3] for s in metrics["Plagiarism-diff"].columns}
    df_plagiarism_diff = pd.DataFrame(d)
    df_plagiarism_diff.to_csv(plot_dir + "/overall_plagiarism_diff.csv", index=False)


def get_packed_metrics(overall_metric_dirs: List[str], mutation):
    """
    :overall_metric_dirs: list of file names of pickles with the individual evaluations.
    :mutation: type of mutation (add or add_sub)
    :return: dict with keys "Style", "Musicality" and "Plagiarism" with their respective DataFrames to plot the heatmaps.
    The value of "Style" is a dictionary of DataFrames with the original styles as keys.
    """
    files = [f for d in overall_metric_dirs for f in glob.glob(os.path.join(d, f'overall_metrics_dict-{mutation}*'))]
    dicts_overall_metrics = [load_pickle(f) for f in files]

    styles = np.unique([d["orig"] for d in dicts_overall_metrics])

    # Packing musicality and plagiarism evaluation
    packed_metrics_aux = {"Style": {target: {orig: 0 for orig in styles} for target in styles},
                          "Musicality": {target: {orig: 0 for orig in styles} for target in styles},
                          "Plagiarism-dist": {target: {orig: 0 for orig in styles} for target in styles},
                          "Plagiarism-diff": {target: {orig: 0 for orig in styles} for target in styles}}

    for d in dicts_overall_metrics:
        packed_metrics_aux["Musicality"][d["target"]][d['orig']] = d['Musicality']
        packed_metrics_aux["Plagiarism-dist"][d["target"]][d['orig']] = d['Plagiarism-dist']
        packed_metrics_aux["Plagiarism-diff"][d["target"]][d['orig']] = d['Plagiarism-diff']
        packed_metrics_aux["Style"][d["target"]][d['orig']] = d['Style']["joined"][d['target']]

    musicality, plagiarism_dist, plagiarism_diff, style_eval = {}, {}, {}, {}
    for target in styles:
        musicality[target] = []
        plagiarism_dist[target] = []
        plagiarism_diff[target] = []
        style_eval[target] = []
        musicality['original'] = []
        plagiarism_dist['original'] = []
        plagiarism_diff['original'] = []
        style_eval['original'] = []

        for orig in styles:
            musicality[target].append(packed_metrics_aux["Musicality"][target][orig])
            plagiarism_dist[target].append(packed_metrics_aux["Plagiarism-dist"][target][orig])
            plagiarism_diff[target].append(packed_metrics_aux["Plagiarism-diff"][target][orig])
            style_eval[target].append(packed_metrics_aux["Style"][target][orig])

            musicality['original'].append(orig)
            plagiarism_dist['original'].append(orig)
            plagiarism_diff['original'].append(orig)
            style_eval['original'].append(orig)

    # Packing style evaluation
    # for d in dicts_overall_metrics:
    #     target = d['target']
    #     style_eval['original'].append(d['orig'])
    #     style_eval[target].append(d["Style"][target])

    packed_metrics = {"Musicality": pd.DataFrame(musicality).set_index('original'),
                      "Plagiarism-dist": pd.DataFrame(plagiarism_dist).set_index('original'),
                      "Plagiarism-diff": pd.DataFrame(plagiarism_diff).set_index('original'),
                      "Style": pd.DataFrame(style_eval).set_index('original')
                      }

    return packed_metrics
