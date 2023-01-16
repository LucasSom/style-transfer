from evaluation.metrics.intervals import get_style_intervals_bigrams_avg
from evaluation.metrics.rhythmic_bigrams import get_style_rhythmic_bigrams_avg


class Style:
    def __init__(self, name, df_char, df):
        self.name = name
        self.embedding = df_char['Embedding'].values[0]
        self.intervals_distribution = get_style_intervals_bigrams_avg(df, name)
        self.rhythmic_bigrams_distribution = get_style_rhythmic_bigrams_avg(df, name)
