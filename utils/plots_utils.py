import copy
import os
from typing import List, Union, Dict

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from matplotlib.ticker import PercentFormatter
from sklearn.manifold import TSNE

import model.colab_tension_vae.params as params
from evaluation.metrics.intervals import matrix_of_adjacent_intervals
from evaluation.metrics.rhythmic_bigrams import matrix_of_adjacent_rhythmic_bigrams
from model.embeddings.style import Style
from utils.files_utils import data_path


def save_plot(plot_dir, plot_name, title=None):
    if not title is None: plt.title(title)
    if not os.path.isdir(plot_dir + "/plots/"):
        os.makedirs(plot_dir + "/plots/")
    print(f"Saving plot as {plot_dir}/plots/{plot_name}.png")
    plt.savefig(f"{plot_dir}/plots/{plot_name}.png")


def plot_area(area, color):
    plt.axvspan(xmin=area[0], xmax=area[1], facecolor=color, alpha=0.3)


def plot_metric(callbacks, epoca_final, metric: str, figsize=(20, 10)):
    plt.figure(figsize=figsize)
    for k, v in callbacks.items():
        if metric in k:
            plt.plot(v, label=k)
    plt.legend()
    plt.savefig(data_path + f'logs/{params.config.time_step / 16}bars_{epoca_final}epochs_{metric}.png')


def plot_train(callbacks, epoca_final, figsize=(20, 10)):
    plot_metric(callbacks, epoca_final, 'loss', figsize)
    plot_metric(callbacks, epoca_final, 'accuracy', figsize)


def calculate_TSNEs(df, column_discriminator=None, space_column='Embedding', n_components=2) -> List[TSNE]:
    """
    :param df: DataFrame with embeddings of the songs.
    :param column_discriminator: Name of column to use as discriminator of subdatasets.
    :param space_column: Name of column where are the embeddings.
    :param n_components: Number of t-SNEs components.
    :return: A list with the t-SNEs for each subdataset.
    """
    # Separamos los subdatasets para cada subplot
    embeddings = [np.vstack(df[space_column].values)]  # dataset completo
    if column_discriminator is not None:
        df.sort_values(by=[column_discriminator], inplace=True)
        for subcase in df[column_discriminator].drop_duplicates():
            embeddings.append(np.vstack((df[df[column_discriminator] == subcase])[space_column].values))

    # Armamos el t-SNE para cada dataset
    return [TSNE(n_components).fit_transform(style_emb) for style_emb in embeddings]


def plot_tsnes_comparison(df, tsne_ds, plot_path, column_discriminator='Style', plot_name='tsne_comparison', style=None,
                          markers=None):
    """
    :param df: pandas dataset
    :param tsne_ds: must have elements of same size
    :param plot_path: directory where to save the plot
    :param column_discriminator: name of column to compare
    :param plot_name: file name where to save the plot
    :param style: `style` parameter of seaborn relplot
    :param markers: `markers` parameter of seaborn relplot
    """

    tsne_result_merged_df = copy.copy(df)
    tsne_result_merged_df['dim_1'] = tsne_ds[:, 0]
    tsne_result_merged_df['dim_2'] = tsne_ds[:, 1]

    sns.relplot(x='dim_1', y='dim_2', hue='Title', data=tsne_result_merged_df, kind='scatter', height=6,
                col=column_discriminator, style=style, markers=markers)
    # lim = (tsne_result.min()-5, tsne_result.max()+5)

    save_plot(plot_path, plot_name)


def plot_tsne(df, tsnes, plot_path, plot_name='tsne', style=None):
    # Plot the result of our TSNE with the label color coded
    tsne_df = copy.copy(df)
    tsne_df['dim_1'] = tsnes[:, 0]
    tsne_df['dim_2'] = tsnes[:, 1]

    grid = sns.relplot(x='dim_1', y='dim_2', hue='Style', data=tsne_df, kind='scatter', height=6, style=style)

    save_plot(plot_path, plot_name)
    return grid

def plot_embeddings(df: pd.DataFrame, emb_column: Union[str, List[str]], emb_styles, plot_dir: str, plot_name="embeddings",
                    include_songs=True):
    # columns_to_plot = ["Style", emb_column] if type(emb_column) is str else ["Style"] + emb_column
    df_tsne = df[["Style", emb_column]]
    df_tsne["Type"] = df.shape[0] * ["Fragment"]

    for style_name, style_emb in emb_styles.items():
        style_row = {"Style": [style_name], emb_column: [style_emb], "Type": ["Style"]}
        df_tsne = pd.concat([df_tsne, pd.DataFrame(style_row)])

    df_tsne = df_tsne if include_songs else df_tsne[df_tsne["Type"] == "Style"]
    embeddings = list(df_tsne[emb_column])
    for i, x in enumerate(embeddings):
        embeddings[i] = np.hstack(x)

    tsne: np.ndarray = TSNE(n_components=2).fit_transform(embeddings)
    grid = plot_tsne(df_tsne, tsne, plot_dir, plot_name, style=("Type" if include_songs else None))
    return grid


def plot_tsne_distributions(tsne_df, plot_dir, plot_name, style_plot=None):
    intervals_tsne: np.ndarray = TSNE(n_components=2).fit_transform(list(tsne_df['intervals_distribution']))
    rhythmic_tsne: np.ndarray = TSNE(n_components=2).fit_transform(list(tsne_df['rhythmic_bigrams_distribution']))

    tsne_df['intervals_dim_1'] = intervals_tsne[:, 0]
    tsne_df['intervals_dim_2'] = intervals_tsne[:, 1]
    tsne_df['rhythmic_dim_1'] = rhythmic_tsne[:, 0]
    tsne_df['rhythmic_dim_2'] = rhythmic_tsne[:, 1]

    sns.relplot(x='intervals_dim_1', y='intervals_dim_2', hue='Name', data=tsne_df, kind='scatter', height=6, style=style_plot)
    save_plot(plot_dir, plot_name + "-intervals", "Intervals distribution")

    sns.relplot(x='rhythmic_dim_1', y='rhythmic_dim_2', hue='Name', data=tsne_df, kind='scatter', height=6, style=style_plot)
    save_plot(plot_dir, plot_name + "-rhythmic_bigrams", "Rhythmic bigrams distribution")


def plot_characteristics_distributions(styles: Dict[str, Style], plot_dir: str, plot_name: str):
    """
    Generate 2 t-SNEs plots of the styles characteristic distributions: one with the intervals distribution and the
    other with the rhythmic bigrams distribution.

    Parameters
    ----------
    styles : Dictionary that maps style names (strings) with its correspondent Style object.

    plot_dir : Directory where to save the generated plots.

    plot_name : File name prefixes. It will be added suffixes "-intervals.png" and "-rhythmic_bigrams.png" to each plot

    """
    tsne_df = {"Name": [], "Style": []}
    for n, s in styles.items():
        tsne_df["Name"].append(n)
        tsne_df["Style"].append(s)
    tsne_df = pd.DataFrame(tsne_df)

    tsne_df['intervals_distribution'] = tsne_df.apply(lambda row: np.hstack(row["Style"].intervals_distribution), axis=1)
    tsne_df['rhythmic_bigrams_distribution'] = tsne_df.apply(lambda row: np.hstack(row["Style"].rhythmic_bigrams_distribution), axis=1)

    plot_tsne_distributions(tsne_df, plot_dir, plot_name)


def plot_fragments_distributions(df: pd.DataFrame, styles: Dict[str, Style], plot_dir: str, plot_name: str):
    """
    Generate 2 tSNEs plots of the fragments and styles characteristic distributions: one with the intervals
    distribution and the other with the rhythmic bigrams distribution.

    Parameters
    ----------
    df : DataFrame
        It has to have columns "Style", "m" and "m'"

    styles : dictionary(k: string, v: Style object)
        Dictionary that maps style names (strings) with its correspondent Style object.

    plot_dir : string
        Directory where to save the generated plots.

    plot_name : string
        File name prefixes. It will be added suffixes "-intervals.png" and "-rhythmic_bigrams.png" to each plot

    """
    tsne_df = {"Name": [], "Type": [], "intervals_distribution": [], "rhythmic_bigrams_distribution": []}

    for _, r in df.iterrows():
        tsne_df["Name"] += ["m", "m'"]
        tsne_df["Type"] += ["fragment", "fragment"]
        tsne_df["intervals_distribution"].append(np.hstack(matrix_of_adjacent_intervals(r["roll"])[0]))
        tsne_df["intervals_distribution"].append(np.hstack(matrix_of_adjacent_intervals(r["NewRoll"])[0]))
        tsne_df["rhythmic_bigrams_distribution"].append(np.hstack(matrix_of_adjacent_rhythmic_bigrams(r["roll"])[0]))
        tsne_df["rhythmic_bigrams_distribution"].append(np.hstack(matrix_of_adjacent_rhythmic_bigrams(r["NewRoll"])[0]))

    for n, s in styles.items():
        tsne_df["Name"].append(n)
        tsne_df["Type"].append("style")
        tsne_df["intervals_distribution"].append(np.hstack(s.intervals_distribution))
        tsne_df["rhythmic_bigrams_distribution"].append(np.hstack(s.rhythmic_bigrams_distribution))

    tsne_df = pd.DataFrame(tsne_df)

    plot_tsne_distributions(tsne_df, plot_dir, plot_name, style_plot="Type")



def bigrams_plot(df, order: List, eval_dir, plot_name, context='talk'):
    if len(order) == 2:
        col = [order[0]]
        row = [order[1]]
        orig, dest = order
    else:
        col = row = order
        orig = dest = 'all'

    sns.set_theme()
    sns.set_context(context)

    sns.displot(data=df, x="value", hue="type", kind='kde', col_order=col, row_order=row)
    plot_area((0, 1), 'C0')
    plot_area((-1, 0), 'C1')

    save_plot(eval_dir, plot_name, f'{plot_name} distribution of \n{orig} transformed to {dest}')


def plagiarism_plot(df, s1, s2, by_distance, eval_dir, context):
    kind = "Distance" if by_distance else "Differences"

    sns.set_theme()
    sns.set_context(context)

    for orig, dest in [(s1, s2), (s2, s1)]:
        sns.displot(data=df[df["Style"] == orig],
                    x=f"{kind} relative ranking",
                    row="target",
                    aspect=2, kind='hist', stat='proportion', bins=np.arange(0, 1.1, 0.1)
                    ).set(title=f"Original style: {orig}\nTarget: {dest}")

        plt.gca().yaxis.set_major_formatter(PercentFormatter(1))

        plot_name = f"plagiarism_{'dist' if by_distance else 'diff'}_{orig}_to_{dest}.png"
        title = f"Place on plagiarism {'dist' if by_distance else 'diff'} ranking from {orig} to {dest}"
        save_plot(eval_dir, plot_name, title)