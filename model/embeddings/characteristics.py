from typing import Tuple, Dict

import dfply
import pandas as pd

from model.embeddings.style import Style
from model.embeddings.embeddings import obtain_std, obtain_embeddings, embeddings_to_rolls


def calculate_characteristics(df) -> Dict[str, Style]:
    df_char = (df
              >> dfply.group_by("Style")
              >> dfply.summarise(Embedding=dfply.X['Embedding'].mean(), Sigma=obtain_std(dfply.X['Embedding']))
              )

    styles = set(df["Style"])

    characteristic_vectors = {
        s: Style(s, df_char[(df_char["Style"] == s)], df)
        for s in styles
    }

    return characteristic_vectors


def obtain_characteristics(df, vae) -> Tuple[pd.DataFrame, Dict[str, Style]]:
    df_emb = obtain_embeddings(df, vae, inplace=True)
    return df_emb, calculate_characteristics(df_emb)


def interpolate_centroids(styles, vae, audio_path):
    d = {s.name.split('/')[-1]: [s.embedding] for s in styles}
    interpolated_styles = []

    for s1 in styles:
        interpolated_styles.append(s1)
        for s2 in styles:
            if not s2 in interpolated_styles:
                s50_emb = (s1.embedding + s2.embedding) / 2
                s25_emb = (s1.embedding + s50_emb) / 2
                s75_emb = (s2.embedding + s50_emb) / 2

                d[f"{s1.name.split('/')[-1]}_25_{s2.name.split('/')[-1]}"] = [s25_emb]
                d[f"{s1.name.split('/')[-1]}_50_{s2.name.split('/')[-1]}"] = [s50_emb]
                d[f"{s1.name.split('/')[-1]}_75_{s2.name.split('/')[-1]}"] = [s75_emb]

    df = pd.DataFrame(d).T.rename(columns={0: 'Embedding'})
    new_rolls = embeddings_to_rolls(df["Embedding"], df.index, "", vae, sparse=False, audio_path=audio_path, save_midi=True,
                                    verbose=False)
    df["New"] = new_rolls

    return df
