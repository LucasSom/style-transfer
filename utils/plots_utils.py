import copy
import os
import os.path

import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE

import model.colab_tension_vae.params as params
from utils.files_utils import data_path


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


def calculate_TSNEs(df, column_discriminator=None, space_column='Embedding', n_components=2):
    # Separamos los subdatasets para cada subplot
    subdatasets = [np.vstack(df[space_column].values)]  # dataset completo
    if column_discriminator is not None:
        df.sort_values(by=[column_discriminator], inplace=True)
        for subcaso in df[column_discriminator].drop_duplicates():
            subdatasets.append(np.vstack((df[df[column_discriminator] == subcaso])[space_column].values))

    # Armamos el t-SNE para cada dataset
    return [TSNE(n_components).fit_transform(ds) for ds in subdatasets]


def plot_tsnes_comparison(df, tsne_ds_list, column_discriminator='Style', path=None, inplace=False):
    """
    :param df: pandas dataset
    :param tsne_ds_list: must have elements of same size
    :param column_discriminator: name of column to compare
    :param inplace: whether to add tsne dimensions to df
    """
    df['dim_1'] = np.concatenate([tr[:, 0] for tr in tsne_ds_list])
    df['dim_2'] = np.concatenate([tr[:, 1] for tr in tsne_ds_list])

    sns.relplot(x='dim_1', y='dim_2', hue='Title', data=df, kind='scatter', height=6, col=column_discriminator)

    plt.show()
    if path is not None:
        plt.savefig(os.path.join(path, "tsne_comparison.png"))

    if not inplace:
        df.drop(columns=['dim_1', 'dim_2'])


def plot_tsne(df, tsne_ds, path=None, inplace=False):
    # Plot the result of our TSNE with the label color coded
    df['dim_1'] = tsne_ds[:, 0]
    df['dim_2'] = tsne_ds[:, 1]

    sns.relplot(x='dim_1', y='dim_2', hue='Title', style='Style', data=df, kind='scatter', height=6)

    plt.show()
    if path is not None:
        plt.savefig(os.path.join(path, "tsne.png"))

    if not inplace:
        df.drop(columns=['dim_1', 'dim_2'])
    # lim = (tsne_result.min()-5, tsne_result.max()+5)


def intervals_talk_plot(merged_df, originals, subplot=0):
    """
    :param merged_df: dataframe
    :param originals: style subsets
    :param subplot: on how many subsets want to divide to plot. If 0, no subplot (ie, do the entire plot)
    """
    sns.set_theme()
    sns.set_context('talk')

    if subplot:
        intervals_plot(merged_df, rows=originals, columns=originals[:subplot], aspect=2)
        intervals_plot(merged_df, rows=originals, columns=originals[subplot:], aspect=2)
    else:
        intervals_plot(merged_df, rows=originals, columns=originals, aspect=2)

    plt.savefig(os.path.join(data_path, "debug_outputs", f"intervals_plot-challenge{subplot}.png"))


def intervals_plot(merged_df, rows, columns, aspect=1):
    sns.displot(data=merged_df,
                col="target",
                row="orig",
                x="value",
                hue="type",
                kind="kde",
                col_order=columns,
                row_order=rows,
                aspect=aspect)
