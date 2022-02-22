import copy
from typing import List

import dfply
import numpy as np
import pandas as pd

from model.colab_tension_vae import util
from roll.guoroll import GuoRoll


def obtain_embeddings(df, vae):
    df_emb = df.groupby('Titulo').sample()
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


def obtain_characteristics(df):
    df_titulo_car = (df
                     >> dfply.group_by('Titulo', 'Autor')
                     >> dfply.summarise(Embedding=dfply.X['Embedding'].mean(), Sigma=obtain_std(dfply.X['Embedding']))
                     )

    df_autor_car = (df
                    >> dfply.group_by('Autor')
                    >> dfply.summarise(Embedding=dfply.X['Embedding'].mean(), Sigma=obtain_std(dfply.X['Embedding']))
                    )

    df_emb_car = pd.concat([
        df >> dfply.mutate(Tipo='Fragmento'),
        df_titulo_car >> dfply.mutate(Tipo='Titulo'),
        df_autor_car >> dfply.mutate(Tipo='Autor', Titulo=dfply.X['Autor'])
    ])

    # save_pickle(df_emb_car, 'df_emb_car')

    def limpiar_columnas(df):
        columnas_nan = set()
        for c in df.columns:
            for s in df[c]:
                if type(s) == float and np.isnan(s):
                    columnas_nan.add(c)

        df.drop(columns=columnas_nan)
        return columnas_nan

    df_caracteristicos = (df_emb_car
                          >> dfply.mask(dfply.X['Tipo'] != 'Fragmento')
                          )

    autores = set(df_emb_car['Autor'])

    caracteristicos_de_autores = {
        autor: df_emb_car[
            (df_emb_car['Tipo'] == 'Autor') &
            (df_emb_car['Titulo'] == autor)
            ]['Embedding'].values[0]
        for autor in autores
    }

    return df_emb_car, limpiar_columnas(df_caracteristicos), caracteristicos_de_autores


def cambiar_estilo(df, caracteristicos, original, objetivo, sample=1, escala=1):
    v_original = caracteristicos[original]
    v_goal = caracteristicos[objetivo]

    return (df
            >> dfply.mask(dfply.X['Autor'] == original, dfply.X['Tipo'] == 'Fragmento')
            >> dfply.group_by('Titulo', 'Autor')
            # >> dfply.sample(sample)
            >> dfply.mutate(Mutacion_add=dfply.X['Embedding'].apply(lambda e: e + v_goal * escala))
            >> dfply.mutate(Mutacion_add_sub=dfply.X['Embedding'].apply(lambda e: e - v_original * escala))
            )


@dfply.make_symbolic
def embeddings_to_rolls(embeddings, vae) -> List[GuoRoll]:
    decoded_matrices = decode_embeddings(embeddings, vae)

    matrices = matrix_sets_to_matrices(decoded_matrices)
    rolls = [GuoRoll(m) for m in matrices]

    return rolls


# Antiguo nombre: get_roll_midi_df
def get_embeddings_roll_df(df_in, vae, column='Embedding', inline=False):
    df = df_in if inline else copy.deepcopy(df_in)

    if type(column) == list:
        for c in column:
            df = get_embeddings_roll_df(df_in, vae, column=c, inline=inline)
        return df

    rolls = embeddings_to_rolls(df[column], vae)
    df[column + 'Roll'] = rolls

    return df


def matrix_sets_to_matrices(matrix_sets: List):
    matrices = []
    for new_matrix_set in matrix_sets:
        new_matrix = np.array([np.hstack(x) for x in zip(*new_matrix_set)])
        sampled_matrix = util.result_sampling(new_matrix)
        sampled_matrix = np.reshape(sampled_matrix, (-1, sampled_matrix.shape[-1]))
        matrices.append(sampled_matrix)
    return matrices
