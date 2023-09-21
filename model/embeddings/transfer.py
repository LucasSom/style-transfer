import dfply
import pandas as pd

from model.embeddings.embeddings import get_embeddings_roll_df


def transfer_style_to(df, vae, model_name: str, characteristics: dict, original_style: str, target_style: str, alphas,
                      mutations, sparse) -> pd.DataFrame:
    df_transferred = transform_embeddings(df, characteristics, original_style, target_style, alphas)

    return get_embeddings_roll_df(df_transferred, vae, model_name, sparse=sparse, save_midi=False,
                                  column=mutations, inplace=True)


def transform_embeddings(df, characteristics: dict, original: str, target: str, alphas, sample=1):
    v_original = characteristics[original].embedding
    v_goal = characteristics[target].embedding

    df = (df
          >> dfply.mask(dfply.X['Style'] == original)
          >> dfply.group_by('Title', 'Style')
          >> dfply.sample(sample, random_state=42)
          )

    for a in alphas:
        df[f"Mutation_add_{a}"] = [r["Embedding"] + v_goal * a for _, r in df.iterrows()]
        df[f"Mutation_add_sub_{a}"] = [r[f"Mutation_add_{a}"] - v_original * a for _, r in df.iterrows()]

    return df
