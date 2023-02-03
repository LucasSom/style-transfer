import pandas as pd
from pandas import DataFrame
from scipy.stats import entropy

from evaluation.metrics.intervals import get_style_intervals_bigrams_avg
from evaluation.metrics.rhythmic_bigrams import get_style_rhythmic_bigrams_avg


def styles_bigrams_entropy(df) -> DataFrame:
    def calculate_entropy(df_style, style, interval):
        probabilities = get_style_intervals_bigrams_avg(df_style, style) if interval else get_style_rhythmic_bigrams_avg(df_style, style)
        return entropy(probabilities, axis=0)

    d = {"Style": [], "Melodic entropy": [], "Rhythmic entropy": []}
    for style in set(df["Style"]):
        d["Style"].append(style)
        d["Melodic entropy"].append(calculate_entropy(df[df["Style"] == style], style, True))
        d["Rhythmic entropy"].append(calculate_entropy(df[df["Style"] == style], style, False))

    return pd.DataFrame(d)
