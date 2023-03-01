from evaluation.metrics.intervals import get_intervals_distribution
from evaluation.metrics.rhythmic_bigrams import get_rhythmic_distribution


class Style:
    def __init__(self, name, df_char, df):
        self.name = name
        if df_char is not None:
            self.embedding = df_char['Embedding'].values[0]
        self.intervals_distribution = get_intervals_distribution(df[df['Style'] == name])
        self.rhythmic_bigrams_distribution = get_rhythmic_distribution(df[df['Style'] == name])
