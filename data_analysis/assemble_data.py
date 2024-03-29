import numpy as np
import ot
import pandas as pd
from scipy.special import rel_entr

from evaluation.metrics.intervals import matrix_of_adjacent_intervals
from evaluation.metrics.rhythmic_bigrams import matrix_of_adjacent_rhythmic_bigrams, possible_patterns
from utils.utils import normalize


def get_df_bigram_matrices(df):
    df["Melodic bigram matrix"] = df.apply(lambda row: matrix_of_adjacent_intervals(row["roll"])[0], axis=1)
    df["Rhythmic bigram matrix"] = df.apply(lambda row: matrix_of_adjacent_rhythmic_bigrams(row["roll"])[0], axis=1)
    return df

def calculate_long_df(df, df_test, styles_train):
    def distanceF(style, part, method_f, row):
        if part == 'Melodic':
            return method_f(style.intervals_distribution, row['Melodic bigram matrix'], True)
        if part == 'Rhythmic':
            return method_f(style.rhythmic_bigrams_distribution, row['Rhythmic bigram matrix'], False)
        # Joined
        return (method_f(style.intervals_distribution, row['Melodic bigram matrix'], True) +
                    method_f(style.rhythmic_bigrams_distribution, row['Rhythmic bigram matrix'], False))

    rolls_long_df = pd.DataFrame.from_records(
        dict(
            target=target,
            method=method_name,
            part=part,
            distance=distanceF(styles_train[target], part, method_f, r),
            **r
        )
        for idx, r in df_test.iterrows()
        for method_name, method_f in dict(linear=linear_distance,
                                          kl=kl,
                                          probability=belonging_probability,
                                          optimal_transport=optimal_transport).items()
        for target in set(df['Style'])
        for part in ['Melodic', 'Rhythmic', 'Joined']
    )
    return rolls_long_df


def calculate_closest_styles(df_test, styles_train, only_ot=False, only_proba=False) -> pd.DataFrame:
    df_test = get_df_bigram_matrices(df_test)

    if not only_ot and not only_proba:
        df_test["Rhythmic closest style (linear)"] = df_test.apply(
            lambda row: rhythmic_closest_style(row["Rhythmic bigram matrix"], styles_train), axis=1)
        df_test["Melodic closest style (linear)"] = df_test.apply(
            lambda row: melodic_closest_style(row["Melodic bigram matrix"], styles_train), axis=1)
        df_test["Joined closest style (linear)"] = df_test.apply(
            lambda row: joined_closest_style(row["Melodic bigram matrix"], row["Rhythmic bigram matrix"], styles_train),
            axis=1)

        df_test["Rhythmic closest style (kl)"] = df_test.apply(
            lambda row: rhythmic_closest_style(row["Rhythmic bigram matrix"], styles_train, method='kl'), axis=1)
        df_test["Melodic closest style (kl)"] = df_test.apply(
            lambda row: melodic_closest_style(row["Melodic bigram matrix"], styles_train, method='kl'), axis=1)
        df_test["Joined closest style (kl)"] = df_test.apply(
            lambda row: joined_closest_style(row["Melodic bigram matrix"], row["Rhythmic bigram matrix"], styles_train,
                                             method='kl'), axis=1)
    if not only_ot:
        df_test["Rhythmic closest style (probability)"] = df_test.apply(
            lambda row: rhythmic_closest_style(row["Rhythmic bigram matrix"], styles_train, method='probability'), axis=1)
        df_test["Melodic closest style (probability)"] = df_test.apply(
            lambda row: melodic_closest_style(row["Melodic bigram matrix"], styles_train, method='probability'), axis=1)
        df_test["Joined closest style (probability)"] = df_test.apply(
            lambda row: joined_closest_style(row["Melodic bigram matrix"], row["Rhythmic bigram matrix"], styles_train,
                                             method='probability'), axis=1)

    if not only_proba:
        df_test["Rhythmic closest style (ot)"] = df_test.apply(
            lambda row: rhythmic_closest_style(row["Rhythmic bigram matrix"], styles_train, method='ot'), axis=1)
        df_test["Melodic closest style (ot)"] = df_test.apply(
            lambda row: melodic_closest_style(row["Melodic bigram matrix"], styles_train, method='ot'), axis=1)
        df_test["Joined closest style (ot)"] = df_test.apply(
            lambda row: joined_closest_style(row["Melodic bigram matrix"], row["Rhythmic bigram matrix"], styles_train,
                                             method='ot'), axis=1)


    return df_test


def kl(P, Q, melodic=True):
    return sum(sum(rel_entr(normalize(P), Q)))


def linear_distance(P, Q, melodic=True):
    return sum(sum(abs(normalize(P) - Q)))


def belonging_probability(M, x, melodic=True):
    return sum(sum(np.log(M) * x))


def optimal_transport(h1, h2, melodic):
    a, b, D = histograms_and_distance(h1, h2, melodic)
    return ot.emd2(a, b, D)


def rhythmic_closest_style(bigram_matrix, styles, method='linear'):
    if method == 'kl':
        min_style_idx = np.argmin(
            [kl(style.rhythmic_bigrams_distribution, bigram_matrix) for style in styles.values()]
        )
    elif method == 'linear':
        min_style_idx = np.argmin(
            [linear_distance(style.rhythmic_bigrams_distribution, bigram_matrix) for style in styles.values()]
        )
    elif method == 'probability':
        min_style_idx = np.argmin(
            [belonging_probability(style.rhythmic_bigrams_distribution, bigram_matrix) for style in styles.values()]
        )
    elif method == 'ot':
        min_style_idx = np.argmin(
            [optimal_transport(style.rhythmic_bigrams_distribution, bigram_matrix, False) for style in styles.values()]
        )
    else:
        raise ValueError(f"{method} is not a valid method.")
    return list(styles.items())[min_style_idx][0]


def melodic_closest_style(bigram_matrix, styles, method='linear'):
    if method == 'kl':
        min_style_idx = np.argmin(
            [kl(style.intervals_distribution, bigram_matrix) for style in styles.values()]
        )
    elif method == 'linear':
        min_style_idx = np.argmin(
            [linear_distance(style.intervals_distribution, bigram_matrix) for style in styles.values()]
        )
    elif method == 'probability':
        min_style_idx = np.argmin(
            [belonging_probability(style.intervals_distribution, bigram_matrix) for style in styles.values()]
        )
    elif method == 'ot':
        min_style_idx = np.argmin(
            [optimal_transport(style.intervals_distribution, bigram_matrix, True) for style in styles.values()]
        )
    else:
        raise ValueError(f"{method} is not a valid method.")
    return list(styles.items())[min_style_idx][0]


def joined_closest_style(interval_matrix, rhythmic_matrix, styles, method='linear'):
    if method == 'kl':
        min_style_idx = np.argmin(
            [abs(kl(style.intervals_distribution, interval_matrix))
             + abs(kl(style.rhythmic_bigrams_distribution, rhythmic_matrix))
             for style in styles.values()]
        )
    elif method == 'linear':
        min_style_idx = np.argmin(
            [abs(linear_distance(style.intervals_distribution, interval_matrix))
             + abs(linear_distance(style.rhythmic_bigrams_distribution, rhythmic_matrix))
             for style in styles.values()]
        )
    elif method == 'probability':
        min_style_idx = np.argmin(
            [abs(belonging_probability(style.intervals_distribution, interval_matrix))
             + abs(belonging_probability(style.rhythmic_bigrams_distribution, rhythmic_matrix))
             for style in styles.values()]
        )
    elif method == 'ot':
        min_style_idx = np.argmin(
            [abs(optimal_transport(style.intervals_distribution, interval_matrix, True))
             + abs(optimal_transport(style.rhythmic_bigrams_distribution, rhythmic_matrix, False))
             for style in styles.values()]
        )
    else:
        raise ValueError(f"{method} is not a valid method.")
    return list(styles.items())[min_style_idx][0]


def histograms_and_distance(h1, h2, melodic=True):
    x = np.arange(-12, 13) if melodic else np.arange(possible_patterns)
    y = np.arange(-12, 13) if melodic else np.arange(possible_patterns)
    x_mesh, y_mesh = np.meshgrid(x, y)
    M = np.dstack((x_mesh, y_mesh))
    M = M.reshape(25 * 25, 2) if melodic else M.reshape(possible_patterns * possible_patterns, 2)
    D = ot.dist(M)

    a, b = np.hstack(h1) / np.sum(h1), np.hstack(h2) / np.sum(h2)
    return a, b, D
