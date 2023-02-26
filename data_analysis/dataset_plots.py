import os

import dfply as dfp
import numpy as np
import ot
import ot.plot
import seaborn as sns
from matplotlib import pyplot as plt

from evaluation.metrics.intervals import get_style_intervals_bigrams_sum
from evaluation.metrics.rhythmic_bigrams import get_style_rhythmic_bigrams_sum, possible_patterns

from utils.plots_utils import save_plot


def plot_styles_bigrams_entropy(entropies, plot_dir, plot_name="styles_complexity"):
    sns.scatterplot(data=entropies, x="Melodic entropy", y="Rhythmic entropy", hue="Style")
    save_plot(plot_dir, plot_name, "Styles entropies for melody and rhythm")


def plot_styles_heatmaps(df, plot_dir):
    histograms = {}
    for style in set(df["Style"]):
        melodic_hist, m_xedges, m_yedges = get_style_intervals_bigrams_sum(np.zeros((25,25)), df[df['Style'] == style])
        plt.imshow(melodic_hist, interpolation='nearest', origin='lower',
                   extent=[m_xedges[0], m_xedges[-1], m_yedges[0], m_yedges[-1]])
        save_plot(plot_dir + "/melodic", f"{style}-melodic", f"Melodic distribution of {style}")

        rhythmic_hist, rx, ry = get_style_rhythmic_bigrams_sum(np.zeros((16,16)), df[df['Style'] == style])
        plt.imshow(rhythmic_hist, interpolation='nearest', origin='lower',
                   extent=[rx[0], rx[-1], ry[0], ry[-1]])
        save_plot(plot_dir + "/rhythmic", f"{style}-rhythmic", f"Rhythmic distribution of {style}")


        histograms[style] = {"melodic_hist": melodic_hist, "rhythmic_hist": rhythmic_hist}
    return histograms


def histograms_and_distance(h1, h2, melodic=True):
    x = np.arange(-12, 13) if melodic else np.arange(possible_patterns)
    y = np.arange(-12, 13) if melodic else np.arange(possible_patterns)
    x_mesh, y_mesh = np.meshgrid(x, y)
    M = np.dstack((x_mesh, y_mesh))
    M = M.reshape(25 * 25, 2) if melodic else M.reshape(possible_patterns * possible_patterns, 2)
    D = ot.dist(M)

    a, b = np.hstack(h1) / np.sum(h1), np.hstack(h2) / np.sum(h2)
    return a, b, D

def plot_heatmap_differences(df, histograms, plot_dir, melodic=True):
    for s1 in set(df["Style"]):
        for s2 in set(df["Style"]):
            if s1 != s2:
                a, b, D = histograms_and_distance(s1, s2, histograms)
                G0 = ot.emd(a, b, D)

                plt.figure(3, figsize=(10, 10))
                title = f'Melodic OT matrix G0 between {s1} and {s2}'
                ot.plot.plot1D_mat(a, b, G0, title)
                save_plot(plot_dir, f"melodic_diff-{s1}_{s2}", title)

                # TODO: sacar para ritmos
                # TODO: en realidad quiero el valor
                
def heatmap_style_differences(diff_table, plot_dir):
    # TODO: sacar para ritmos
    diff_table.to_csv(os.path.join(plot_dir, f"melodic_diff.csv"))
    sns.heatmap(diff_table.pivot(values='d', index='s1', columns='s2'), annot=True)
    save_plot(plot_dir, 'melodic_diff')


def plot_closeness(df, orig, dest, eval_path, context='talk'):
    fig = plt.figure(figsize=(24, 18))
    sns.set_theme()
    sns.set_context(context)
    title = f"Closest styles of {orig} rolls" if dest == 'nothing' else f"Closest styles of {orig} rolls to {dest}"
    fig.suptitle(title)

    i = 1
    for kind in ['Rhythmic', 'Melodic', 'Joined']:
        for method in ['linear', 'kl', 'probability', 'ot']:
            ax = fig.add_subplot(3, 4, i)
            plt.hist(df[f"{kind} closest style ({method})"])
            ax.title.set_text(f"{kind} closest style ({method})")
            i += 1

    save_plot(eval_path, f"closest_styles-{orig}_to_{dest}", "Joined closest style (ot)")
    plt.close()


def plot_closest_ot_style(df, eval_path, context='talk'):
    """
    :param df: DataFrame with column 'Closest style (ot)'
    """
    for s in set(df["Style"]):
        fig = plt.figure(figsize=(18, 18))
        title = f"Closest styles of test {s} rolls"
        fig.suptitle(title)
        sns.set_theme()
        sns.set_context(context)

        ax1 = fig.add_subplot(1, 3, 1)
        plt.hist(df[df["Style"] == s]['Melodic closest style (ot)'])
        ax1.title.set_text("Melodic closest style (ot)")

        ax2 = fig.add_subplot(1, 3, 2)
        plt.hist(df[df["Style"] == s]['Rhythmic closest style (ot)'])
        ax2.title.set_text("Rhythmic closest style (ot)")

        ax3 = fig.add_subplot(1, 3, 3)
        plt.hist(df[df["Style"] == s]['Joined closest style (ot)'])
        ax3.title.set_text("Joined closest style (ot)")

        save_plot(eval_path, f"closest_styles_ot-{s}", 'Joined closest style (ot)')


def plot_distances_distribution(df, eval_path, context='talk', by_style=True, single_plot=False):
    if single_plot:
        rolls_long_ot_df = (df
                >> dfp.gather('distance_type', 'distance', dfp.contains('to'))
                >> dfp.mutate(
            target=dfp.X.distance_type.apply(lambda x: x.split(' ')[3]),
            distance_metric=dfp.X.distance_type.apply(lambda x: x.split(' ')[1]),
            distance_type2=dfp.X.distance_type.apply(lambda x: x.split(' ')[0])
            )
        )
        sns.displot(data=rolls_long_ot_df, x='distance', hue='target', col='distance_type2',
                    row='Style', kind='kde')
        save_plot(eval_path, "OT distrbution")
    else:
        for orig in set(df["Style"]):
            fig = plt.figure(figsize=(40, 10))
            title = f"Closest styles of test {orig} rolls"
            fig.suptitle(title)
            sns.set_theme()
            sns.set_context(context)

            if by_style:
                for i, s2 in enumerate(set(df["Style"])):
                    ax = fig.add_subplot(1, 4, i + 1)
                    sns.kdeplot(df[df["Style"] == orig][f'Melodic ot to {s2}'])
                    sns.kdeplot(df[df["Style"] == orig][f'Rhythmic ot to {s2}'])
                    sns.kdeplot(df[df["Style"] == orig][f'Joined ot to {s2}'])

                    plt.legend(labels=[f"Melodic ot to {s2}", f"Rhythmic ot to {s2}", f'Joined ot to {s2}'])
                    ax.title.set_text(f"Distribution of distances to {s2}")

                save_plot(eval_path, f"ot_distances_style-{orig}", f'Distribution of distances of {orig} rolls to {s2}')
            else:
                for i, kind in enumerate(["Melodic", "Rhythmic", "Joined"]):
                    ax = fig.add_subplot(1, 3, i + 1)
                    sns.kdeplot(df[df["Style"] == orig][f'{kind} ot to Bach'])
                    sns.kdeplot(df[df["Style"] == orig][f'{kind} ot to ragtime'])
                    sns.kdeplot(df[df["Style"] == orig][f'{kind} ot to Mozart'])
                    sns.kdeplot(df[df["Style"] == orig][f'{kind} ot to Frescobaldi'])

                    plt.legend(labels=[f'{kind} ot to Bach', f'{kind} ot to ragtime', f'{kind} ot to Mozart', f'{kind} ot to Frescobaldi'])
                    ax.title.set_text(f"Distribution of {kind} distances")
                save_plot(eval_path, f"ot_distances_kind-{orig}", f"Distribution of {kind} distances")


def plot_accuracy(df, eval_path):
    df = df[df["method"] == 'optimal_transport']
    df = df[df["part"] == 'Joined']
    df = df[df["Style"] == df["target"]]

    d = {}
    for s in set(df["Style"]):
        sub_df = df[df["Style"] == s]
        acc = sub_df[sub_df['Joined closest style (ot)'] == s].shape[0] / sub_df.shape[0]
        d[s] = acc

    plt.bar(d.keys(), d.values())
    save_plot(eval_path, 'styles_accuracy', 'Proportion of rolls that are classified on its own style')
