import pytest
from tensorflow import keras

from dodo import analyze_training, preprocessed_data, do_embeddings, models
from model.embeddings.embeddings import obtain_embeddings
from utils.files_utils import load_pickle, preprocessed_data_path, path_saved_models, get_embedding_path, \
    get_reconstruction_path, get_characteristics_path, get_emb_path, data_path


@pytest.fixture
def brmf4_prep():
    return load_pickle(file_name=preprocessed_data_path+"bach-rag-moz-fres-4")



@pytest.fixture
def brmf4_emb():
    return load_pickle(file_name=get_embedding_path('brmf_4b'))


@pytest.fixture
def model_name():
    return "bach_rag_moz_fres"


'''
# CÁLCULO DE EMBEDDINGS

from model.embeddings import obtain_embeddings

df_emb = obtain_embeddings(df_midi, vae)
df_emb
'''


def test_obtain_embeddings(brmf4_prep):
    vae = keras.models.load_model(path_saved_models)

    df_emb = obtain_embeddings(brmf4_prep, vae)

    assert "Embedding" in df_emb.columns
    for e in df_emb.Embedding:
        assert e.shape == (96,)


def test_analyze_training(brmf4_emb):
    model_name = 'brmf_4b'
    analyze_training(df_path=preprocessed_data(4), model_name=model_name, b=4,
                     targets=get_reconstruction_path(model_name))


def test_characteristics(brmf4_emb):
    model_name = '4-small_br'
    model_path = data_path + f'/{model_name}/vae'
    do_embeddings(preprocessed_data(4), model_path, model_path, get_characteristics_path(model_name),
                  get_emb_path(model_name), 4)

def test_all_models_characteristics(brmf4_emb):
    import h5py
    print(h5py.__version__)

    for model_name in models:
        model_path = data_path + f'/{model_name}/vae'
        do_embeddings(preprocessed_data(4), model_path, model_path, get_characteristics_path(model_name),
                      get_emb_path(model_name), 4)
