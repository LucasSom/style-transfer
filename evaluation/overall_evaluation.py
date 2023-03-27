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

    # Plot heatmap musicality
    sns.heatmap(metrics["Musicality"], annot=True, fmt='g', vmin=0, vmax=100)
    save_plot(plot_dir, "heatmap_musicality", "Avg of % of permutations\nthat are less musical than the generated")

    # Plot heatmap plagiarism
    sns.heatmap(metrics["Plagiarism"], annot=True, fmt='g', vmin=0, vmax=1)
    save_plot(plot_dir, "heatmap_plagiarism", "Avg of % of rolls\nthat are more different than the generated")
