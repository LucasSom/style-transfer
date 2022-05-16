import pytest
from tensorflow import keras

from model.embeddings.embeddings import obtain_embeddings
from utils.files_utils import load_pickle, data_path, preprocessed_data_path


@pytest.fixture
def brmf_prep():
    return load_pickle(file_name=preprocessed_data_path+"bach_rag_moz_fres")


@pytest.fixture
def model_name():
    return "bach_rag_moz_fres"


'''
# CÁLCULO DE EMBEDDINGS

from model.embeddings import obtain_embeddings

df_emb = obtain_embeddings(df_midi, vae)
df_emb
'''


def test_obtain_embeddings(brmf_prep, model_name):
    vae = keras.models.load_model(data_path + f"saved_models/{model_name}/")

    df_emb = obtain_embeddings(brmf_prep, vae)

    assert "Embedding" in df_emb.columns
    for e in df_emb.Embedding:
        assert e.shape == (96,)


'''
# VECTORES CARACTERÍSTICOS

from utils.files_utils  import datasets_name
from model.embeddings import obtain_characteristics

df_emb_car, df_caracteristicos, caracteristicos_de_autores = obtain_characteristics(df_emb)

caracteristicos_pkl_name = 'df_car'+datasets_name(songs)
save_pickle(df_caracteristicos, caracteristicos_pkl_name)
save_pickle(caracteristicos_de_autores, 'caracteristicos_de_autores')
'''

'''
# TRANSFERIR ESTILOS

from utils.utils import exp_disponibles
from model.embeddings import transform_embeddings, get_embeddings_roll_df

#@title ¿Ya reconstruye? ¿Transferimos estilos o solo probamos la reconstrucción?
transfer = False #@param {type:"boolean"}

df_transfered = transform_embeddings(df_emb_car, caracteristicos_de_autores, ds_original, ds_objetivo, escala=1) if transfer else df_emb_car
# save_pickle(df_transfered, nombre_pickle)


# TODO: Llegué hasta acá refactorizando
exp_list = [e for e in exp_disponibles(df_transfered) if 'roll' not in e]
print("Lista de experimentos:")
print(exp_list)

for exp in exp_list:
  get_embeddings_roll_df(df_transfered, vae, column=exp, inline=True)

save_pickle(df_transfered, nombre_pickle)
print("Guardado en " + nombre_pickle + ".pkl")
'''
