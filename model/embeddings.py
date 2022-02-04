import dfply
import numpy as np
import pandas as pd


def obtain_embeddings(df, vae):
    df_emb = df.groupby('Titulo').sample()
    # df_sampled['Embedding'].iloc[0][0]

    # df_emb = df
    t = vae.get_layer(name='encoder')(np.stack(df_emb['roll']))
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
