import copy
from collections import Counter
from typing import List, Iterable

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
        df = df.groupby('Style').sample(n=samples, random_state=42)
        df_emb = df
    else:
        df_emb = df.groupby('Style').sample(n=samples, random_state=42)
    # df_sampled['Embedding'].iloc[0][0]

    roll_matrices = [r.matrix.todense() if r.sparse else r.matrix for r in df_emb['roll']]
    t = vae.get_layer(name='encoder')(np.stack(np.array(roll_matrices)))
    df_emb['Embedding'] = list(t.numpy())
    return df_emb


def decode_embeddings(embeddings, model):
    decoder = model.get_layer(name='decoder')
    return [decoder((np.expand_dims(e, 0))) for e in embeddings]


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
            >> dfply.mutate(Mutation_add=dfply.X['Embedding'].apply(lambda e: e + v_goal * scale))
            >> dfply.mutate(Mutation_add_sub=dfply.X['Mutation_add'].apply(lambda e: e - v_original * scale))
            )


@dfply.make_symbolic
def embeddings_to_rolls(embeddings: Iterable, roll_names: List[str], suffix: str, model, sparse: bool, audio_path: str,
                        save_midi: bool, verbose=False) -> List[GuoRoll]:
    decoded_matrices = decode_embeddings(embeddings, model)

    matrices = matrix_sets_to_matrices(decoded_matrices)
    rolls = [GuoRoll(m, f"{r_name}-{suffix}", sparse=sparse, audio_path=audio_path, save_midi=save_midi, verbose=verbose)
             for m, r_name in zip(matrices, roll_names)]

    return rolls


def get_embeddings_roll_df(df_in: pd.DataFrame, model, model_name: str, sparse: bool, save_midi: bool,
                           column='Embedding', inplace=False) -> pd.DataFrame:
    df = df_in if inplace else copy.deepcopy(df_in)

    if type(column) == list:
        for c in column:
            df = get_embeddings_roll_df(df_in, model, model_name, sparse, save_midi, column=c, inplace=True)
        return df

    audio_path = get_audios_path(model_name)
    roll_names = [r.name for r in df['roll']]
    rolls = embeddings_to_rolls(df[column], roll_names, column, model, sparse, audio_path, save_midi)
    df[f"{column}-NewRoll"] = rolls

    return df


def matrix_sets_to_matrices(matrix_sets: list):
    matrices = []
    for new_matrix_set in matrix_sets:
        new_matrix = np.array([np.hstack(x) for x in zip(*new_matrix_set)])
        sampled_matrix = util.result_sampling(new_matrix)
        sampled_matrix = np.reshape(sampled_matrix, (-1, sampled_matrix.shape[-1]))
        matrices.append(sampled_matrix)
    return matrices


def get_reconstruction(df_emb, model, model_name: str):
    get_embeddings_roll_df(df_emb, model, model_name, sparse=True, save_midi=False, inplace=True)
    df_emb.rename(columns={'Embedding-NewRoll': 'Reconstruction'}, inplace=True)
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


def interpolate(model, embedding1: np.array, embedding2: np.array, name1: str, name2: str, audios_path: str,
                alpha: float, save_midi=True, sparse=False) -> List[GuoRoll]:
    subtitle = f"{name1}_{name2}_"
    interpolation = alpha * embedding1 + (1 - alpha) * embedding2

    return embeddings_to_rolls([embedding1, interpolation, embedding2],
                               [subtitle + '0', subtitle + f'{alpha * 100}', subtitle + '100'],
                               "", model, sparse, audios_path + 'song_interpolations/', save_midi)
