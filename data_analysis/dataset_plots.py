import os

import pandas as pd
import dfply as dfp
import numpy as np
import ot
import ot.plot
import seaborn as sns
from matplotlib import pyplot as plt

from data_analysis.assemble_data import histograms_and_distance
from evaluation.metrics.intervals import get_style_intervals_bigrams_sum
from evaluation.metrics.rhythmic_bigrams import get_style_rhythmic_bigrams_sum

from utils.plots_utils import save_plot


def plot_styles_bigrams_entropy(entropies, plot_dir, plot_name="styles_complexity"):
    sns.scatterplot(data=entropies, x="Melodic entropy", y="Rhythmic entropy", hue="Style")
    save_plot(plot_dir, plot_name, "Styles entropies for melody and rhythm")


def plot_styles_heatmaps_and_get_histograms(df, plot_dir):
    """
    For each style, plots the characteristic heatmap and calculates the melodic and rhythmic characteristic matrix

    :param df: DataFrame with columns 'Style' and 'roll'.
    :param plot_dir: directory where save the plots.
    :return: A dictionary that maps style names to another dictionary with maps 'melodic_hist' and 'rhythmic_hist' with
    its corresponding histograms.
    """
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


def plot_closeness(df, orig, dest, eval_path, context='talk', only_joined_ot=False):
    fig = plt.figure(figsize=(24, 18))
    sns.set_theme()
    sns.set_context(context)
    title = f"Closest styles of {orig} rolls" if dest == 'nothing' else f"Closest styles of {orig} rolls to {dest}"
    fig.suptitle(title)

    if only_joined_ot:
        df = df[df['target'] == dest]
        plt.hist(df["Joined closest style (ot)"])
    else:
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
                for i, part in enumerate(["Melodic", "Rhythmic", "Joined"]):
                    ax = fig.add_subplot(1, 3, i + 1)
                    sns.kdeplot(df[df["Style"] == orig][f'{part} ot to Bach'])
                    sns.kdeplot(df[df["Style"] == orig][f'{part} ot to ragtime'])
                    sns.kdeplot(df[df["Style"] == orig][f'{part} ot to Mozart'])
                    sns.kdeplot(df[df["Style"] == orig][f'{part} ot to Frescobaldi'])

                    plt.legend(labels=[f'{part} ot to Bach', f'{part} ot to ragtime', f'{part} ot to Mozart', f'{part} ot to Frescobaldi'])
                    ax.title.set_text(f"Distribution of {part} distances")
                save_plot(eval_path, f"ot_distances_kind-{orig}", f"Distribution of {part} distances")


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


def plot_musicality_distribution(dfs: dict, eval_path, plot_suffix='', context='talk', only_probability=False,
                                 only_joined=True):

    methods = ['probability'] if only_probability else ['linear', 'kl', 'ot', 'probability']
    parts = ['Joined'] if only_joined else ["Melodic", "Rhythmic", "Joined"]

    for method in methods:
        for i, part in enumerate(parts):
            sns.set_context(context)
            plt.figure(figsize=(10, 6))
            title = f"{part} musicality ({method})"
            sns.set_theme()

            for df in dfs.values():
                sns.kdeplot(df[f'{part} musicality difference ({method})'])

            plt.legend(labels=dfs.keys())
            save_plot(eval_path, f'{part}_musicality_{method}{plot_suffix}', title + plot_suffix)


def plot_accuracy_distribution(dfs_test_path, eval_dir):
    """
    For each method and part, it plots a Box plot of the accuracies of belonging to the correct style
    """
    cat_long = pd.concat([
        pd.read_pickle(f'{dfs_test_path}{i}.pkl')
        >> dfp.mutate(fold=i)
        for i in range(5)
    ])

    @dfp.make_symbolic
    def find_closest(targets_distances):
        return targets_distances.sort_values('distance').iloc[0]['target']

    cat_long >> dfp.drop(dfp.contains('matrix'))

    closest_df = (
        cat_long
        >> dfp.drop(dfp.contains('matrix'))
        >> dfp.group_by('method', 'part', 'roll_id', 'Title', 'fold')
        >> dfp.summarize(
            closest=find_closest(dfp.X),
            style=dfp.X.Style.iloc[0]
        )
    )

    @dfp.make_symbolic
    def matches(df):
        return df['style'] == df['closest']

    accuracy_df = (
        closest_df
        >> dfp.mutate(matches=matches(dfp.X))
        >> dfp.group_by('fold', 'method', 'part', 'style')
        >> dfp.summarize(accuracy=dfp.X.matches.mean())
    )

    sns.catplot(data=accuracy_df, x='style', y='accuracy', row='method',
                col='part', kind='box')
    save_plot(eval_dir, 'style_closeness_accuracy', 'Styles closeness accuracy')

    return cat_long, closest_df, accuracy_df


def plot_styles_confusion_matrix(df, styles, plot_path):
    d = {s_y: [df[(df["Style"] == s_x) & (df["Joined closest style (ot)"] == s_y)].shape[0] for s_x in styles] for s_y in styles}
    d['Style'] = list(styles)
    sns.heatmap(pd.DataFrame(d).set_index('Style'), annot=True, fmt='d')
    save_plot(plot_path, 'confusion_matrix', 'Confusion matrix of original style and classified style')
