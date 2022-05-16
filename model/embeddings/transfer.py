import pandas as pd

from model.embeddings.embeddings import transform_embeddings, get_embeddings_roll_df


def transfer_style_to(df, vae, characteristics: dict, original_style: str, target_style: str) -> pd.DataFrame:
    df_transfered = transform_embeddings(df, characteristics, original_style, target_style, scale=1)

    return get_embeddings_roll_df(df_transfered, vae, name_new_column=f"{original_style}2{target_style}", inplace=True)
