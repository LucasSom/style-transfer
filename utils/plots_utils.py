import copy
import os
from typing import List, Union, Dict

import dfply
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.ticker import PercentFormatter
from sklearn.manifold import TSNE

import model.colab_tension_vae.params as params
from evaluation.metrics.intervals import matrix_of_adjacent_intervals
from evaluation.metrics.rhythmic_bigrams import matrix_of_adjacent_rhythmic_bigrams
from model.embeddings.embeddings import decode_embeddings, matrix_sets_to_matrices, get_accuracy, get_accuracies
from model.embeddings.style import Style
from utils.files_utils import data_path


def save_plot(plot_dir, plot_name, title=None):
    """
    Save the plot in a subfolder 'plots' of plot_dir with name 'plot_name and title 'title'.
    """
    title = ""
    plt.title(title) if not title is None else plt.title(plot_name)
    if not os.path.isdir(plot_dir + "/plots/"):
        os.makedirs(plot_dir + "/plots/")
    plt.tight_layout()
    print(f"Saving plot as {plot_dir}/plots/{plot_name}.png")
    plt.savefig(f"{plot_dir}/plots/{plot_name}.png")
    plt.close()


def plot_area(area, color):
    plt.axvspan(xmin=area[0], xmax=area[1], facecolor=color, alpha=0.3)


def plot_metric(callbacks, logs_dir, metric: str, only_general_loss: bool):
    if only_general_loss:
        for k, v in callbacks.items():
            if k in ['loss', 'val_loss']:
                plt.plot(v, label=k)
    else:
        for k, v in callbacks.items():
            if metric in k:
                plt.plot(v, label=k)
    plt.legend()
    save_plot(logs_dir, metric)


def plot_train(callbacks, logs_dir):
    plot_metric(callbacks, logs_dir, 'loss', True)
    plot_metric(callbacks, logs_dir, 'accuracy', False)


def plot_accuracies(df, model, logs_path):
    accuracies = {}
    encode = model.get_layer(name='encoder')

    for s in set(df["Style"]):
        sub_df = df[df["Style"] == s]
        original_matrices = [r.matrix for r in sub_df['roll']]
        decoded_matrices = decode_embeddings(encode(np.stack(original_matrices)).numpy(), model)
        reconstructed_matrices = matrix_sets_to_matrices(decoded_matrices)
        # save_pickle(original_matrices, data_path + "orig_matrices_ragtime")
        mel_acc, mel_rhythm_acc, bass_acc, bass_rhythm_acc = get_accuracies(x=reconstructed_matrices,
                                                                            y=original_matrices)
        # metrics = model.evaluate(x=reconstructed_matrices, y=original_matrices, workers=-1, use_multiprocessing=True)
        # model.metrics_names
        # accuracies[s] = metrics[1]
        accuracies[s] = mel_acc, mel_rhythm_acc, bass_acc, bass_rhythm_acc

    x = np.arange(len(accuracies))  # the label locations
    width = 0.2  # the width of the bars
    multiplier = 0

    fig, ax = plt.subplots()  # layout='constrained')

    for attribute, measurement in accuracies.items():
        offset = width * multiplier
        rects = ax.bar(x + offset, measurement, width, label=attribute)
        ax.bar_label(rects, padding=3)
        multiplier += 1

    ax.set_ylabel('Accuracy')
    ax.set_xticks(x + width, ["melody_pitch", "melody_rhythm", "bass_pitch", "bass_rhythm"])
    ax.legend(loc='upper left', ncols=4)

    # y_pos = np.arange(len(accuracies))
    # plt.bar(y_pos, accuracies.values())
    # plt.xticks(y_pos, accuracies.keys())

    save_plot(logs_path, "style_accuracy", "Reconstruction accuracies for each style")


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


def plot_embeddings(df: pd.DataFrame, emb_column: Union[str, List[str]], emb_styles, plot_dir: str,
                    plot_name="embeddings",
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

    tsne: np.ndarray = TSNE(n_components=2).fit_transform(np.array(embeddings))
    grid = plot_tsne(df_tsne, tsne, plot_dir, plot_name, style=("Type" if include_songs else None))
    return grid


def plot_tsne_distributions(tsne_df, plot_dir, plot_name, style_plot=None):
    intervals_tsne: np.ndarray = TSNE(n_components=2, perplexity=5).fit_transform(
        np.array(tsne_df['intervals_distribution']))
    rhythmic_tsne: np.ndarray = TSNE(n_components=2, perplexity=5).fit_transform(
        np.array(tsne_df['rhythmic_bigrams_distribution']))

    tsne_df['intervals_dim_1'] = intervals_tsne[:, 0]
    tsne_df['intervals_dim_2'] = intervals_tsne[:, 1]
    tsne_df['rhythmic_dim_1'] = rhythmic_tsne[:, 0]
    tsne_df['rhythmic_dim_2'] = rhythmic_tsne[:, 1]

    sns.relplot(x='intervals_dim_1', y='intervals_dim_2', hue='Name', data=tsne_df, kind='scatter', height=6,
                style=style_plot)
    save_plot(plot_dir, plot_name + "-intervals", "Intervals distribution")

    sns.relplot(x='rhythmic_dim_1', y='rhythmic_dim_2', hue='Name', data=tsne_df, kind='scatter', height=6,
                style=style_plot)
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

    tsne_df['intervals_distribution'] = tsne_df.apply(lambda row: np.hstack(row["Style"].intervals_distribution),
                                                      axis=1)
    tsne_df['rhythmic_bigrams_distribution'] = tsne_df.apply(
        lambda row: np.hstack(row["Style"].rhythmic_bigrams_distribution), axis=1)

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


def plagiarism_plot(df, orig, dest, mutation, by_distance, eval_dir, context):
    kind = "Distance" if by_distance else "Differences"
    sns.set_theme(context)

    sns.displot(data=df[df["Style"] == orig],
                x=f"{kind} relative ranking",
                row="target",
                aspect=2, kind='hist', stat='proportion', bins=np.arange(0, 1.1, 0.1)
                ).set(title=f"Original style: {orig}\nTarget: {dest}")

    plt.gca().yaxis.set_major_formatter(PercentFormatter(1))

    plot_name = f"plagiarism_{'dist' if by_distance else 'diff'}_{orig}_to_{dest}-{mutation}.png"
    title = f"Place on plagiarism {'dist' if by_distance else 'diff'} ranking from {orig} to {dest} ({mutation})"
    save_plot(eval_dir, plot_name, title)


def plot_IR_distributions(df: pd.DataFrame, orig, dest, plot_dir):
    for style in set(df["Style"]):
        df_style = df[df["Style"] == style]

        df_permutations = pd.DataFrame()
        for _, row in df_style.iterrows():
            irs = row["IRs perm"]
            df_permutation = pd.DataFrame({"Style": [style for _ in range(len(irs))],
                                           "type": ["IR perm" for _ in range(len(irs))],
                                           "IR": [ir for ir in irs],
                                           })
            df_permutations = pd.concat([df_permutations, df_permutation])

        df_style = (df_style
                    >> dfply.gather("type", "IR", ["IR orig", "IR trans"])
                    )
        df_to_plot = pd.concat([df_style[["Style", "type", "IR"]], df_permutations])

        sns.displot(df_to_plot[["type", "IR"]].reset_index(), x="IR", kind='kde', hue='type', rug=True)
        sns.displot(df_to_plot[["type", "IR"]].reset_index(), x="IR", kind='hist', hue='type', rug=False,
                    stat="probability")
        # sns.displot(df_to_plot[["type", "IR"]], kind="kde", hue='type', rug=True)

        save_plot(plot_dir, f"IR-{style}-{orig}_to_{dest}", f"IR distribution of {style} ({orig} to {dest}) style")


def plot_intervals_improvements(orig, dest, interval_distances, plot_path, context='talk'):
    sns.set_theme()
    sns.set_context(context)
    sns.kdeplot(data=interval_distances, x="log(m's/ms)")
    sns.displot(data=interval_distances, x="log(m's'/ms')", kind="kde")
    plt.title(f'Interval distribution of \n{orig} transformed to {dest}')
    plt.savefig(os.path.join(data_path, plot_path, f"intervals_{orig}_to_{dest}.png"))
