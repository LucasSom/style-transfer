import pandas as pd
import seaborn as sns

from utils.plots_utils import save_plot


def heatmap_style_evaluation(df, orig, plot_dir, context='talk'):
    # sns.set_theme(context)
    sns.heatmap(df, annot=True, fmt='g', vmin=0, vmax=100)
    save_plot(plot_dir, f"heatmap_style_{orig}", f"Avg of % of approaches to styles from {orig}")


def overall_evaluation(metrics, plot_dir, context='talk'):
    """
    :metrics: dict with keys "Style", "Musicality" and "Plagiarism" with their respective DataFrames to plot the heatmaps
    """
    # sns.set_theme(context)

    # Plot heatmap for Style evaluation
    for orig, df in metrics["Style"].items():
        heatmap_style_evaluation(df, orig, plot_dir, context)

    d = {orig: [(s, df[s][s]) for s in df.columns] for orig, df in metrics["Style"].items()}
    df_style = pd.DataFrame(index=metrics["Style"].keys(), columns=metrics["Style"].keys())
    for s in d.keys():
        df_style.at[s, s] = 0
    for orig, l in d.items():
        for dest, value in l:
            df_style.at[orig, dest] = value  # index: orig; columns: dest
    df_style.to_csv(plot_dir + "/overall_style.csv", index=True)

    # Plot heatmap musicality
    sns.heatmap(metrics["Musicality"], annot=True, fmt='g', vmin=0, vmax=100)
    save_plot(plot_dir, "heatmap_musicality", "Avg of % of permutations\nthat are less musical than the generated")
    d = {s: [sum(metrics["Musicality"][s]) / 3] for s in metrics["Musicality"].columns}
    df_mus = pd.DataFrame(d)
    df_mus.to_csv(plot_dir + "/overall_musicality.csv", index=False)

    # Plot heatmap plagiarism
    sns.heatmap(metrics["Plagiarism"], annot=True, fmt='g', vmin=0, vmax=1)
    save_plot(plot_dir, "heatmap_plagiarism", "Avg of % of rolls\nthat are more different than the generated")
    d = {s: [sum(metrics["Plagiarism"][s]) / 3] for s in metrics["Plagiarism"].columns}
    df_plagiarism = pd.DataFrame(d)
    df_plagiarism.to_csv(plot_dir + "/overall_plagiarism.csv", index=False)
