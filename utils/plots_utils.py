import copy
import os
from typing import List

import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from matplotlib.ticker import PercentFormatter
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


def plot_tsnes_comparison(df, tsne_ds_list, column_discriminator='Style'):
    """
    :param df: pandas dataset
    :param tsne_ds_list: must have elements of same size
    :param column_discriminator: name of column to compare
    """
    tsne_result_merged_df = copy.copy(df)

    tsne_result_merged_df['dim_1'] = np.concatenate([tr[:, 0] for tr in tsne_ds_list])
    tsne_result_merged_df['dim_2'] = np.concatenate([tr[:, 1] for tr in tsne_ds_list])

    sns.relplot(x='dim_1', y='dim_2', hue='Title', data=tsne_result_merged_df, kind='scatter', height=6,
                col=column_discriminator)
    # lim = (tsne_result.min()-5, tsne_result.max()+5)


def plot_tsne(df, tsne_ds):
    # Plot the result of our TSNE with the label color coded
    tsne_result_df = copy.copy(df)
    tsne_result_df['dim_1'] = tsne_ds[:, 0]
    tsne_result_df['dim_2'] = tsne_ds[:, 1]

    sns.relplot(x='dim_1', y='dim_2', hue='Title', style='Tipo', data=tsne_result_df, kind='scatter', height=6)
    # lim = (tsne_result.min()-5, tsne_result.max()+5)


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

    sns.displot(data=df, col="target", row="Style", x="value", hue="type", kind='kde', col_order=col, row_order=row)
    # plt.show()

    plot_area((0, 1), 'C0')
    plot_area((-1, 0), 'C1')

    plt.title(f'Interval distribution of \n{orig} transformed to {dest}')
    plt.savefig(os.path.join(data_path, f"debug_outputs/plots/intervals/{orig}_to_{dest}.png"))
    plt.show()


def single_plagiarism_plot(df, context, by_distance):
    kind = "Distance" if by_distance else "Differences"
    s1 = list(set(df["Style"]))[0]
    s2 = list(set(df["Style"]))[1]

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


# def plagiarism_plot(df, order, context, by_distance):
#     if len(order) == 2:
#         col = [order[0]]
#         row = [order[1]]
#         orig, dest = order
#     else:
#         col = row = order
#         orig = dest = 'all'
#
#     sns.set_theme()
#     sns.set_context(context)
#
#     sns.displot(data=df,
#                 col="target",
#                 row="Style",
#                 x="value",
#                 hue="type",
#                 kind='hist',
#                 stat='proportion',
#                 # binwidth=1,
#                 col_order=col,
#                 row_order=row)
#
#     plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
#
#     plt.savefig(os.path.join(data_path, "debug_outputs", f"plagiarism_{'dist' if by_distance else 'diff'}_{orig}_to_{dest}.png"))
#     plt.show()
