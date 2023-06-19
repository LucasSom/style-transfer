import copy
from collections import Counter
from typing import List

import dfply
import numpy as np
import pandas as pd

from model.colab_tension_vae import util, params
from roll.guoroll import GuoRoll
from utils.files_utils import get_audios_path


def obtain_embeddings(df: pd.DataFrame, vae, samples=500, inplace=False) -> pd.DataFrame:
    """
    Takes a DataFrame and a model, and applies the 'encoder' function to a sample of rolls of the df.

    :param df: dataframe with rolls to which encode
    :param vae: trained model
    :param samples: number of fragments to sample for each style
    :param inplace: if True, perform operation in-place.
    :return: the input DataFrame with a new column 'Embedding' with the result of the encoding (ndarrays of shape (96,))
    """
    samples = min(samples, min(Counter(df["Style"]).values()))
    if inplace:
        # TODO (March): Poner seed. Samplear igual cantidad de fragmentos para cada estilo
        df = df.groupby('Style').sample(n=samples, random_state=42)
        df_emb = df
    else:
        # TODO (March): Poner seed. Samplear igual cantidad de fragmentos para cada estilo
        df_emb = df.groupby('Style').sample(n=samples, random_state=42)
    # df_sampled['Embedding'].iloc[0][0]

    t = vae.get_layer(name='encoder')(np.stack([r.matrix for r in df_emb['roll']]))
    df_emb['Embedding'] = list(t.numpy())
    return df_emb


def decode_embeddings(embedding, model):
    decoder = model.get_layer(name='decoder')
    return [decoder((np.expand_dims(e, 0))) for e in embedding]


@dfply.make_symbolic
def obtain_std(embeddings):
    return np.vstack(embeddings).std(axis=0)


def transform_embeddings(df, characteristics: dict, original: str, target: str, scale=1, sample=1):
    v_original = characteristics[original].embedding
    v_goal = characteristics[target].embedding

    return (df
            >> dfply.mask(dfply.X['Style'] == original)
            >> dfply.group_by('Title', 'Style')
            >> dfply.sample(sample)
            >> dfply.mutate(Mutacion_add=dfply.X['Embedding'].apply(lambda e: e + v_goal * scale))
            >> dfply.mutate(Mutacion_add_sub=dfply.X['Embedding'].apply(lambda e: e - v_original * scale))
            )


@dfply.make_symbolic
def embeddings_to_rolls(embeddings, roll_names, suffix, audio_path, model, verbose=False) -> List[GuoRoll]:
    decoded_matrices = decode_embeddings(embeddings, model)

    matrices = matrix_sets_to_matrices(decoded_matrices)
    rolls = [GuoRoll(m, f"{r_name}-{suffix}", audio_path=audio_path, verbose=verbose) for m, r_name in zip(matrices, roll_names)]

    return rolls


def get_embeddings_roll_df(df_in, model, model_name: str, column='Embedding', inplace=False) -> pd.DataFrame:
    name_new_column = "NewRoll"

    df = df_in if inplace else copy.deepcopy(df_in)

    if type(column) == list:
        # TODO(march): esto pisa df constantemente, salvo que se haga inplace
        print("===== PASO POR EL IF DE get_embeddings_roll_df =====")
        for c, n in zip(column, name_new_column):
            df = get_embeddings_roll_df(df_in, model, model_name, column=c, inplace=inplace)
        return df

    new_name_suffix = f"{column}-{name_new_column}"
    audio_path = get_audios_path(model_name)
    roll_names = [r.name for r in df['roll']]
    rolls = embeddings_to_rolls(df[column], roll_names, new_name_suffix, audio_path, model)
    df[name_new_column] = rolls

    return df


def matrix_sets_to_matrices(matrix_sets: list):
    matrices = []
    for new_matrix_set in matrix_sets:
        new_matrix = np.array([np.hstack(x) for x in zip(*new_matrix_set)])
        sampled_matrix = util.result_sampling(new_matrix)
        sampled_matrix = np.reshape(sampled_matrix, (-1, sampled_matrix.shape[-1]))
        matrices.append(sampled_matrix)
    return matrices


def get_reconstruction(df, model, model_name: str, samples, inplace=False):
    df_emb = obtain_embeddings(df, model, samples, inplace)
    get_embeddings_roll_df(df_emb, model, model_name, inplace=True)
    return df_emb


def get_accuracy(x: List[np.array], y: List[np.array]) -> int:
    n = len(x)
    assert len(y) == n
    acc = 0
    N, M = x[0].shape

    for x_i, y_i in zip(x, y):
        acc += sum(sum(x_i == y_i))

    return acc / (N * M * n)


def get_accuracies(x: List[np.array], y: List[np.array]):
    n = len(x)
    assert len(y) == n
    mel_acc, mel_rhythm_acc, bass_acc, bass_rhythm_acc = 0, 0, 0, 0
    N, M = x[0].shape

    for x_i, y_i in zip(x, y):
        x_mel_notes = x_i[:, :params.config.melody_dim]
        x_mel_rhythm = x_i[:, params.config.melody_dim]
        x_bass_notes = x_i[:, params.config.melody_dim + 1 : -1]
        x_bass_rhythm = x_i[:, -1]
        y_mel_notes = y_i[:, :params.config.melody_dim]
        y_mel_rhythm = y_i[:, params.config.melody_dim]
        y_bass_notes = y_i[:, params.config.melody_dim + 1: -1]
        y_bass_rhythm = y_i[:, -1]

        mel_acc += sum(sum((x_mel_notes == y_mel_notes).T) == (params.config.melody_dim * np.ones(params.config.time_step)))
        mel_rhythm_acc += sum(x_mel_rhythm == y_mel_rhythm)
        bass_acc += sum(sum((x_bass_notes == y_bass_notes).T) == (params.config.bass_dim * np.ones(params.config.time_step)))
        bass_rhythm_acc += sum(x_bass_rhythm == y_bass_rhythm)

    return mel_acc / (N * n), mel_rhythm_acc / (N * n), bass_acc / (N * n), bass_rhythm_acc / (N * n)
