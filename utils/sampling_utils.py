import pandas as pd


def sample_uniformly(df, column, n=5):
    """
    Samples n rolls uniformly distributed given the value of the column.
    """
    n = min(n, df.shape[0])
    df = df.sort_values(by=[column]).reset_index()
    df_sampled = pd.DataFrame(columns=df.columns)

    for i in range(0, df.shape[0]-1, int(df.shape[0] / n)):
        df_sampled.loc[df_sampled.shape[0]] = df.loc[i]

    return df_sampled


def balanced_sampling(df, n_samples=500):
    """
    Samples n_samples rolls for each style.

    :param df: DataFrame to sample.
    :param n_samples: Number of samples to sample
    :return: DataFrame with n_samples rolls for each style.
    """
    df_sampled = pd.DataFrame()
    for style in set(df["Style"]):
        sub_df = df[df["Style"] == style].sample(n=n_samples, random_state=24)
        df_sampled = pd.concat([df_sampled, sub_df])

    return df_sampled
