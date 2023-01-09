import pandas as pd

from model.embeddings.embeddings import transform_embeddings, get_embeddings_roll_df


def transfer_style_to(df, vae, model_name: str, characteristics: dict, original_style: str, target_style: str) -> pd.DataFrame:
    df_transferred = transform_embeddings(df, characteristics, original_style, target_style, scale=1)

    return get_embeddings_roll_df(df_transferred, vae, model_name, column="Mutacion_add_sub", inplace=True)
