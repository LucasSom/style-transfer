import pandas as pd
import seaborn as sns
from numpy import float64

from utils.plots_utils import save_plot


# def heatmap_style_evaluation(df, orig, plot_dir, context='talk'):
#     # sns.set_theme(context)
#     sns.heatmap(df, annot=True, fmt='g', vmin=0, vmax=100)
#     save_plot(plot_dir, f"heatmap_style_{orig}", f"Avg of % of approaches to styles from {orig}")


def overall_evaluation(metrics, plot_dir, context='talk'):
    """
    :metrics: dict with keys "Style", "Musicality" and "Plagiarism" with their respective DataFrames to plot the heatmaps
    """
    # sns.set_theme(context)

    # Plot heatmap for Style evaluation
    # for orig, df in metrics["Style"].items():
    #     heatmap_style_evaluation(df, orig, plot_dir, context)

    d = {orig: [(s, df[s][s]) for s in df.columns] for orig, df in metrics["Style"].items()}
    df_style = pd.DataFrame(index=metrics["Style"].keys(), columns=metrics["Style"].keys(), dtype=float64)
    for s in d.keys():
        df_style.at[s, s] = 0.0
    for orig, l in d.items():
        for dest, value in l:
            df_style.at[orig, dest] = value  # index: orig; columns: dest
    df_style.to_csv(plot_dir + "/overall_style.csv", index=True)
    sns.heatmap(df_style, annot=True, annot_kws={'fontsize': 'x-large'}, fmt='g', vmin=0, vmax=100)
    save_plot(plot_dir, f"heatmap_style", f"Promedios de % de acercamiento\na los nuevos estilos")

    # Plot heatmap musicality
    sns.heatmap(metrics["Musicality"], annot=True, annot_kws={'fontsize': 'x-large'}, fmt='g', vmin=0, vmax=100)
    save_plot(plot_dir, "heatmap_musicality", "Promedio de % de permutaciones\nque son menos musicales que el generado")
    d = {s: [sum(metrics["Musicality"][s]) / 3] for s in metrics["Musicality"].columns}
    df_mus = pd.DataFrame(d)
    df_mus.to_csv(plot_dir + "/overall_musicality.csv", index=False)

    # Plot heatmap plagiarism (dist)
    sns.heatmap(metrics["Plagiarism-dist"], annot=True, annot_kws={'fontsize': 'x-large'}, fmt='g', vmin=0, vmax=1)
    save_plot(plot_dir, "heatmap_plagiarism_dist", "Promedio de % de rolls\nque son más diferentes que el generado\n(distancia)")
    d = {s: [sum(metrics["Plagiarism-dist"][s]) / 3] for s in metrics["Plagiarism-dist"].columns}
    df_plagiarism_dist = pd.DataFrame(d)
    df_plagiarism_dist.to_csv(plot_dir + "/overall_plagiarism_dist.csv", index=False)

    # Plot heatmap plagiarism (diff)
    sns.heatmap(metrics["Plagiarism-diff"], annot=True, annot_kws={'fontsize': 'x-large'}, fmt='g', vmin=0, vmax=1)
    save_plot(plot_dir, "heatmap_plagiarism_diff", "Promedio de % de rolls\nque son más diferentes que el generado\n(diferencia)")
    d = {s: [sum(metrics["Plagiarism-diff"][s]) / 3] for s in metrics["Plagiarism-diff"].columns}
    df_plagiarism_diff = pd.DataFrame(d)
    df_plagiarism_diff.to_csv(plot_dir + "/overall_plagiarism_diff.csv", index=False)
