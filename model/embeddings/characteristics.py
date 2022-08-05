from typing import Tuple

import dfply
import pandas as pd

from model.embeddings.embeddings import obtain_std, obtain_embeddings


def obtain_characteristics(df, column='Style') -> dict:
    df_car = (df
              >> dfply.group_by(column)
              >> dfply.summarise(Embedding=dfply.X['Embedding'].mean(), Sigma=obtain_std(dfply.X['Embedding']))
              )

    types = set(df[column])

    characteristic_vectors = {
        t: df_car[(df_car[column] == t)]['Embedding'].values[0]
        for t in types
    }

    return characteristic_vectors  # df_emb_car, limpiar_columnas(df_caracteristicos)


def calculate_characteristics(df, vae, column='Style') -> Tuple[pd.DataFrame, dict]:
    df_emb = obtain_embeddings(df, vae, inplace=True)
    return df_emb, obtain_characteristics(df_emb, column)
