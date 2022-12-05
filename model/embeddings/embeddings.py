import copy
from typing import List

import dfply
import numpy as np
import pandas as pd

from model.colab_tension_vae import util
from roll.guoroll import GuoRoll
from utils.files_utils import get_audios_path


def obtain_embeddings(df: pd.DataFrame, vae, inplace=False) -> pd.DataFrame:
    """
    Takes a DataFrame and a model, and applies the 'encoder' function to all rolls of the df.

    :param df: dataframe with rolls to which encode
    :param vae: trained model
    :param inplace: if True, perform operation in-place.
    :return: the input DataFrame with a new column 'Embedding' with the result of the encoding (ndarrays of shape (96,))
    """
    if inplace:
        # TODO (March): Poner seed
        df = df.groupby('Title').sample()
        df_emb = df
    else:
        # TODO (March): Poner seed
        df_emb = df.groupby('Title').sample()
    # df_sampled['Embedding'].iloc[0][0]

    t = vae.get_layer(name='encoder')(np.stack([r.matrix for r in df_emb['roll']]))
    df_emb['Embedding'] = list(t.numpy())
    return df_emb


def decode_embeddings(embedding, vae):
    decoder = vae.get_layer(name='decoder')
    return [decoder((np.expand_dims(e, 0))) for e in embedding]


@dfply.make_symbolic
def obtain_std(embeddings):
    return np.vstack(embeddings).std(axis=0)


def transform_embeddings(df, characteristics: dict, original: str, target: str, scale=1, sample=1):
    v_original = characteristics[original]
    v_goal = characteristics[target]

    return (df
            >> dfply.mask(dfply.X['Style'] == original)
            >> dfply.group_by('Title', 'Style')
            >> dfply.sample(sample)
            >> dfply.mutate(Mutacion_add=dfply.X['Embedding'].apply(lambda e: e + v_goal * scale))
            >> dfply.mutate(Mutacion_add_sub=dfply.X['Embedding'].apply(lambda e: e - v_original * scale))
            )


@dfply.make_symbolic
def embeddings_to_rolls(embeddings, original_rolls, suffix, audio_path, vae) -> List[GuoRoll]:
    decoded_matrices = decode_embeddings(embeddings, vae)

    matrices = matrix_sets_to_matrices(decoded_matrices)
    rolls = [GuoRoll(m, f"{o_r.name}-{suffix}", audio_path=audio_path) for m, o_r in zip(matrices, original_rolls)]

    return rolls


def get_embeddings_roll_df(df_in, vae, model_name: str, column='Embedding', name_new_column='NewRoll', inplace=False) \
        -> pd.DataFrame:
    df = df_in if inplace else copy.deepcopy(df_in)

    if type(column) == list:
        # TODO(march): esto pisa df constantemente, salvo que se haga inplace
        for c, n in zip(column, name_new_column):
            df = get_embeddings_roll_df(df_in, vae, model_name, column=c, name_new_column=n, inplace=inplace)
        return df

    new_name_suffix = f"{column}-{name_new_column}"
    audio_path = get_audios_path(model_name)
    rolls = embeddings_to_rolls(df[column], df['roll'], new_name_suffix, audio_path, vae)
    df[new_name_suffix] = rolls

    return df


def matrix_sets_to_matrices(matrix_sets: list):
    matrices = []
    for new_matrix_set in matrix_sets:
        new_matrix = np.array([np.hstack(x) for x in zip(*new_matrix_set)])
        sampled_matrix = util.result_sampling(new_matrix)
        sampled_matrix = np.reshape(sampled_matrix, (-1, sampled_matrix.shape[-1]))
        matrices.append(sampled_matrix)
    return matrices


def get_reconstruction(df, model, model_name: str, inplace=False):
    df_emb = obtain_embeddings(df, model, inplace)
    get_embeddings_roll_df(df_emb, model, model_name, inplace=True)
    return df_emb
