import copy
import os
from typing import List

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from matplotlib.ticker import PercentFormatter
from sklearn.manifold import TSNE

import model.colab_tension_vae.params as params
from utils.files_utils import data_path


def save_plot(plot_path, plot_name):
    if not os.path.isdir(plot_path):
        os.makedirs(plot_path)
    plt.savefig(f"{plot_path}/{plot_name}.png")


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
    df['dim_1'] = np.concatenate([tr[:, 0] for tr in tsne_ds])
    df['dim_2'] = np.concatenate([tr[:, 1] for tr in tsne_ds])

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


def plot_characteristics(df: pd.DataFrame, plot_path: str, emb_column: str, emb_styles, plot_name="characteristics"):
    df_tsne = df[["Style", emb_column]]
    df_tsne["Type"] = df.shape[0] * ["Fragment"]

    for style_name, style_emb in emb_styles.items():
        style_row = {"Style": style_name, emb_column: style_emb, "Type": "Style"}
        df_tsne = df_tsne.append(style_row, ignore_index=True)

    embeddings = list(df_tsne[emb_column])
    for i, x in enumerate(embeddings):
        embeddings[i] = np.hstack(x)

    tsne: np.ndarray = TSNE(n_components=2).fit_transform(embeddings)
    grid = plot_tsne(df_tsne, tsne, plot_path, plot_name, style="Type")
    return grid

def plot_area(area, color):
    plt.axvspan(xmin=area[0], xmax=area[1], facecolor=color, alpha=0.3)


def intervals_plot(df, order: List, context='talk'):
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
    # plt.show()

    plot_area((0, 1), 'C0')
    plot_area((-1, 0), 'C1')

    plt.title(f'Interval distribution of \n{orig} transformed to {dest}')
    plt.savefig(os.path.join(data_path, f"debug_outputs/plots/intervals/{orig}_to_{dest}.png"))
    plt.show()


def plagiarism_plot(df, context, s1, s2, by_distance):
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

        plt.savefig(os.path.join(data_path, "debug_outputs/plots/plagiarism/pruebas",
                                 f"plagiarism_{'dist' if by_distance else 'diff'}_{orig}_to_{dest}.png"))
        plt.show()
