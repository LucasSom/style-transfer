import os
from collections import Counter

import pandas as pd
import dfply as dfp
import numpy as np
import ot
import ot.plot
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.stats import kstest

from data_analysis.assemble_data import histograms_and_distance
from evaluation.metrics.intervals import get_style_intervals_bigrams_sum
from evaluation.metrics.rhythmic_bigrams import get_style_rhythmic_bigrams_sum

from utils.plots_utils import save_plot
from utils.utils import normalize


def plot_styles_bigrams_entropy(entropies, plot_dir, plot_name="styles_complexity", english=True):
    if not english:
        entropies = entropies.rename(
            columns={"Melodic entropy": "Entropía melódica", "Rhythmic entropy": "Entropía rítmica"})
    x_label = "Melodic entropy" if english else "Entropía melódica"
    y_label = "Rhythmic entropy" if english else "Entropía rítmica"
    title = "Styles entropies for melody and rhythm" if english else "Entropías de melodía y ritmo para cada estilo"

    sns.scatterplot(data=entropies, x=x_label, y=y_label, hue="Style")
    save_plot(plot_dir, plot_name, title)


def plot_styles_heatmaps_and_get_histograms(df, plot_dir, english=False):
    """
    For each style, plots the characteristic heatmap and calculates the melodic and rhythmic characteristic matrix

    :param df: DataFrame with columns 'Style' and 'roll'.
    :param plot_dir: directory where save the plots.
    :param english: whether annotations are in english
    :return: A dictionary that maps style names to another dictionary with maps 'melodic_hist' and 'rhythmic_hist' with
    its corresponding histograms.
    """
    histograms = {}
    for style in set(df["Style"]):
        melodic_hist, m_xedges, m_yedges = get_style_intervals_bigrams_sum(np.zeros((25, 25)), df[df['Style'] == style])
        sns.heatmap(normalize(melodic_hist), xticklabels=m_xedges, yticklabels=m_yedges, cmap="viridis")
        plt.xlabel('First interval' if english else 'Primer intervalo')
        plt.ylabel('Second interval' if english else 'Segundo intervalo')
        title = f"Melodic distribution of {style}" if english else f"Distribución melódica de {style}"
        save_plot(plot_dir + "/melodic", f"{style}-melodic", title)

        rhythmic_hist, rx, ry = get_style_rhythmic_bigrams_sum(np.zeros((16, 16)), df[df['Style'] == style])
        sns.heatmap(normalize(rhythmic_hist), xticklabels=rx, yticklabels=ry, cmap="viridis")
        plt.xlabel('First rhythmic pattern' if english else 'Primer patrón rítmico')
        plt.ylabel('Second rhythmic pattern' if english else 'Segundo patrón rítmico')
        title = f"Rhythmic distribution of {style}" if english else f"Distribución rítmica de {style}"
        save_plot(plot_dir + "/rhythmic", f"{style}-rhythmic", title)

        histograms[style] = {"melodic_hist": melodic_hist, "rhythmic_hist": rhythmic_hist} # TODO: TENGO QUE NORMALIZAR
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

                # TODO: sacar para ritmos. En realidad quiero el valor


def heatmap_style_differences(diff_table, plot_dir):
    # TODO: sacar para ritmos
    diff_table.to_csv(os.path.join(plot_dir, f"melodic_diff.csv"))
    sns.heatmap(diff_table.pivot(values='d', index='s1', columns='s2'), annot=True)
    save_plot(plot_dir, 'melodic_diff')


def plot_closeness(df, orig, dest, mutation, eval_path, context='talk', only_joined_ot=False, english=False):
    fig = plt.figure(figsize=(5, 5))
    sns.set_theme(context)
    if english:
        title = f"Closest styles of {orig} rolls" if dest == 'nothing' else f"Closest styles of {orig} rolls to {dest}"
    else:
        title = f"Estilos más cercanos de los fragmentos de estilo {orig}" if dest == 'nothing' else f"Estilos más cercanos de los fragmentos de estilo {orig} a {dest}"

    if only_joined_ot:
        if 'target' in df.columns:
            df = df[df['target'] == dest]
        df.sort_values(by='Joined closest style (ot)', inplace=True)
        plt.hist(df["Joined closest style (ot)"])
    else:
        i = 1
        for kind in ['Rhythmic', 'Melodic', 'Joined']:
            for method in ['linear', 'kl', 'probability', 'ot']:
                ax = fig.add_subplot(3, 4, i)
                plt.hist(df[f"{kind} closest style ({method})"])
                ax.title.set_text(f"{kind} closest style ({method})")
                i += 1

    if dest == 'nothing':
        if mutation == 'dataset':
            name = f"closest_styles-{orig}"
        else:
            name = f"closest_styles-{orig}-{mutation}"
    else:
        if mutation == 'dataset':
            name = f"closest_styles-{orig}_to_{dest}"
        else:
            name = f"closest_styles-{orig}_to_{dest}-{mutation}"

    save_plot(eval_path, name, title)
    plt.close()


def plot_distances(distances, orig, dest, kind, mutation, plot_path, context='talk'):
    sns.set_theme(context)

    d = {"style": [], "distance": []}
    for s, ds in distances.items():
        d["style"] += len(ds) * [s]
        d["distance"] += ds
    df = pd.DataFrame(d)

    sns.barplot(data=df, x="style", y="distance", errorbar="sd")

    save_plot(plot_path, f"distances_{orig}_{dest}-{kind}-{mutation}",
              f"Distances to styles\nafter {orig} to {dest} transformation ({kind})")


def plot_closest_ot_style(df, plot_path, context='talk'):
    """
    :param df: DataFrame with column 'Closest style (ot)'
    :param plot_path: path where save the plot
    :param context: seaborn context
    """
    for s in set(df["Style"]):
        fig = plt.figure(figsize=(18, 18))
        title = f"Closest styles of test {s} rolls"
        fig.suptitle(title)
        sns.set_theme(context)

        ax1 = fig.add_subplot(1, 3, 1)
        plt.hist(df[df["Style"] == s]['Melodic closest style (ot)'])
        ax1.title.set_text("Melodic closest style (ot)")

        ax2 = fig.add_subplot(1, 3, 2)
        plt.hist(df[df["Style"] == s]['Rhythmic closest style (ot)'])
        ax2.title.set_text("Rhythmic closest style (ot)")

        ax3 = fig.add_subplot(1, 3, 3)
        plt.hist(df[df["Style"] == s]['Joined closest style (ot)'])
        ax3.title.set_text("Joined closest style (ot)")

        save_plot(plot_path, f"closest_styles_ot-{s}", 'Joined closest style (ot)')

    fig = plt.figure(figsize=(25, 15))
    title = f"Estilo más próximo para los fragmentos de cada estilo"
    fig.suptitle(title)
    sns.set_theme(context)
    for i, s in enumerate(set(df["Style"])):
        ax = fig.add_subplot(1, 4, i + 1)
        c = Counter(df[df["Style"] == s]['Joined closest style (ot)'])
        n = sum(c.values())
        plt.bar(["Mozart", "Bach", "Frescobaldi", "ragtime"],
                [c["Mozart"] / n * 100, c["Bach"] / n * 100, c["Frescobaldi"] / n * 100, c["ragtime"] / n * 100])
        ax.title.set_text(s)
    save_plot(plot_path, f"estilos_mas_proximos", s)


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

                    plt.legend(labels=[f'{part} ot to Bach', f'{part} ot to ragtime', f'{part} ot to Mozart',
                                       f'{part} ot to Frescobaldi'])
                    ax.title.set_text(f"Distribution of {part} distances")
                save_plot(eval_path, f"ot_distances_kind-{orig}", f"Distribution of {part} distances")


def plot_accuracy(df, eval_path):
    sns.set_theme(context='talk')
    if "method" in df.columns:
        df = df[df["method"] == 'optimal_transport']
    if "part" in df.columns:
        df = df[df["part"] == 'Joined']
    if "target" in df.columns:
        df = df[df["Style"] == df["target"]]

    d = {}
    for s in set(df["Style"]):
        sub_df = df[df["Style"] == s]
        acc = sub_df[sub_df['Joined closest style (ot)'] == s].shape[0] / sub_df.shape[0]
        d[s] = acc

    plt.bar(d.keys(), d.values())
    save_plot(eval_path, 'styles_accuracy', 'Proportion of rolls that are classified on its own style')


def plot_musicality_distribution(dfs: dict, eval_path, plot_suffix='', context='talk', only_probability=False,
                                 only_joined=True, english=False):
    if english:
        methods = ['probability'] if only_probability else ['linear', 'kl', 'ot', 'probability']
        parts = ['Joined'] if only_joined else ["Melodic", "Rhythmic", "Joined"]
    else:
        methods = ['probabilidad'] if only_probability else ['lineal', 'kl', 'ot', 'probabilidad']
        parts = ['conjunta'] if only_joined else ["melódica", "rítmica", "conjunta"]

    for method in methods:
        for i, part in enumerate(parts):
            sns.set_context(context)
            plt.figure(figsize=(10, 6))
            title = f"{part} musicality ({method})" if english else f"Musicalidad {part} ({method})"
            sns.set_theme()

            for df in dfs.values():
                part_name = part if english else 'Joined'
                method_name = method if english else \
                    ('probability' if method == 'probabilidad' else
                     ('linear' if method == 'lineal' else method))

                sns.kdeplot(df[f'{part_name} musicality difference ({method_name})'])

            plt.xlabel("Difference with musicality distribution"
                       if english else "Diferencia con la distribución de musicalidad")
            plt.ylabel("Density" if english else "Densidad")
            plt.legend(labels=dfs.keys() if english else ['train', 'test', 'permutaciones'])
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
    d = {s_y: [df[(df["Style"] == s_x) & (df["Joined closest style (ot)"] == s_y)].shape[0] for s_x in styles] for s_y
         in styles}
    d['Style'] = list(styles)
    m = pd.DataFrame(d).set_index('Style')
    sns.heatmap(m, annot=True, fmt='d')
    m.to_csv(plot_path + '/confusion_matrix.csv')
    print(f"Saving confusion matrix as {plot_path + '/confusion_matrix.csv'}")
    save_plot(plot_path, 'confusion_matrix', 'Confusion matrix of original style and classified style')
