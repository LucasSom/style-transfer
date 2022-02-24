import copy

import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE

from model.colab_tension_vae import params as guo_params
from utils.files_utils import data_path


def plot_train(callbacks, epoca_final):
    plt.figure(figsize=(10, 10))
    for k, v in callbacks.items():
        if 'loss' in k:
            plt.plot(v, label=k)
    plt.legend()
    plt.savefig(data_path + f'logs/{guo_params.time_step / 16}bars_{epoca_final}epochs.png')


def calculate_TSNEs(df, column_discriminator=None, space_column='Embedding', n_components=2):
    # Separamos los subdatasets para cada subplot
    subdatasets = [np.vstack(df[space_column].values)]  # dataset completo
    if column_discriminator is not None:
        df.sort_values(by=[column_discriminator], inplace=True)
        for subcaso in df[column_discriminator].drop_duplicates():
            subdatasets.append(np.vstack((df[df[column_discriminator] == subcaso])[space_column].values))

    # Armamos el t-SNE para cada dataset
    return [TSNE(n_components).fit_transform(ds) for ds in subdatasets]


def plot_tsnes_comparison(df, tsne_ds_list, column_discriminator='Autor'):
    """
    :param df: pandas dataset
    :param tsne_ds_list: must have elements of same size
    :param column_discriminator: name of column to compare
    """
    tsne_result_merged_df = copy.copy(df)

    tsne_result_merged_df['dim_1'] = np.concatenate([tr[:, 0] for tr in tsne_ds_list])
    tsne_result_merged_df['dim_2'] = np.concatenate([tr[:, 1] for tr in tsne_ds_list])

    sns.relplot(x='dim_1', y='dim_2', hue='Titulo', data=tsne_result_merged_df, kind='scatter', height=6,
                col=column_discriminator)
    # lim = (tsne_result.min()-5, tsne_result.max()+5)


def plot_tsne(df, tsne_ds):
    # Plot the result of our TSNE with the label color coded
    tsne_result_df = copy.copy(df)
    tsne_result_df['dim_1'] = tsne_ds[:, 0]
    tsne_result_df['dim_2'] = tsne_ds[:, 1]

    sns.relplot(x='dim_1', y='dim_2', hue='Titulo', style='Tipo', data=tsne_result_df, kind='scatter', height=6)
    # lim = (tsne_result.min()-5, tsne_result.max()+5)
