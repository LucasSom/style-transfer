import os

import numpy as np
import ot
import ot.plot
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from evaluation.metrics.intervals import get_style_intervals_bigrams_sum
from evaluation.metrics.rhythmic_bigrams import get_style_rhythmic_bigrams_sum

from utils.plots_utils import save_plot


def plot_styles_bigrams_entropy(entropies, plot_dir, plot_name="styles_complexity"):
    sns.scatterplot(data=entropies, x="Melodic entropy", y="Rhythmic entropy", hue="Style")
    save_plot(plot_dir, plot_name, "Styles entropy for melody and rhythm")


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


def histograms_and_distance(s1, s2, histograms):
    x = np.arange(-12, 13)
    y = np.arange(-12, 13)
    x_mesh, y_mesh = np.meshgrid(x, y)
    M = np.dstack((x_mesh, y_mesh))
    M = M.reshape(25 * 25, 2)
    D = ot.dist(M)
    a, b = histograms[s1]["melodic_hist"], histograms[s2]["melodic_hist"]
    a, b = np.hstack(a) / np.sum(a), np.hstack(b) / np.sum(b)
    return a, b, D

def plot_heatmap_differences(df, histograms, plot_dir):
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
                
def heatmap_differences_table(df, histograms, plot_dir):
    # TODO: sacar para ritmos
    def style_diff(s1, s2):
        if s1 != s2:
            a, b, D = histograms_and_distance(s1, s2, histograms)
            return ot.emd2(a, b, D)
        return 0

    diff_table = pd.DataFrame(
        {
            's1': s1,
            's2': s2,
            'd': style_diff(s1, s2)
        }
        for s1 in set(df['Style'])
        for s2 in set(df['Style'])
    )

    diff_table.to_csv(os.path.join(plot_dir, f"melodic_diff.csv"))
    sns.heatmap(diff_table.pivot(values='d', index='s1', columns='s2'), annot=True)
    save_plot(plot_dir, 'melodic_diff')