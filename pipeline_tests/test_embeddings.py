import numpy as np
import pandas as pd
import pytest
from tensorflow import keras

from dodo import analyze_training, preprocessed_data, do_embeddings, models
from model.colab_tension_vae.params import init
from model.embeddings.embeddings import obtain_embeddings
from model.embeddings.style import Style
from utils.files_utils import load_pickle, preprocessed_data_path, path_saved_models, get_embedding_path, \
    get_reconstruction_path, get_characteristics_path, get_emb_path, data_path
from utils.plots_utils import plot_distributions


@pytest.fixture
def brmf4_prep():
    return load_pickle(file_name=preprocessed_data_path+"bach-rag-moz-fres-4")



@pytest.fixture
def brmf4_emb():
    return load_pickle(file_name=get_embedding_path('brmf_4b'))


@pytest.fixture
def model_name():
    return "bach_rag_moz_fres"

@pytest.fixture
def characteristics():
    init(4)
    df_char = pd.DataFrame({
        "Style": ["Pepe", "Jose", "Baile"],
        "Embedding": [np.zeros(96), np.zeros(96), np.ones(96)],
        "Sigma": [np.zeros(96), np.zeros(96), np.ones(96)],
    })

    df = load_pickle(f'{data_path}4-small_br/embeddings/df_emb.pkl')
    df["Style"][:2] = "Pepe"
    df["Style"][2:4] = "Jose"
    df["Style"][4:] = "Baile"

    styles = {
        "Pepe": Style("Pepe", df_char=df_char, df=df),
        "Jose": Style("Jose", df_char=df_char, df=df),
        "Baile": Style("Baile", df_char=df_char, df=df)
    }

    styles["Pepe"].intervals_distribution = np.ones((24,24))
    styles["Jose"].intervals_distribution = np.ones((24,24))
    styles["Jose"].intervals_distribution[10,11] += 1

    styles["Baile"].intervals_distribution = np.zeros((24,24))
    styles["Baile"].intervals_distribution[10:13,10:13] += 2

    return styles


def test_obtain_embeddings(brmf4_prep):
    vae = keras.models.load_model(path_saved_models)

    df_emb = obtain_embeddings(brmf4_prep, vae)

    assert "Embedding" in df_emb.columns
    for e in df_emb.Embedding:
        assert e.shape == (96,)


def test_analyze_training():
    model_name = 'brmf_4b'
    analyze_training(df_path=preprocessed_data(4), model_name=model_name, b=4,
                     targets=get_reconstruction_path(model_name))


def test_plot_distributions(characteristics):
    plot_dir = data_path + 'debug_outputs/'
    plot_name = "distributions"
    plot_distributions(characteristics, plot_dir, plot_name)


def test_characteristics(): # TODO: correr este test para ver cómo inicializo Style
    model_name = '4-small_br'
    model_path = data_path + f'/{model_name}/vae'
    do_embeddings(preprocessed_data(4), model_path, model_path, get_characteristics_path(model_name),
                  get_emb_path(model_name), 4)

def test_all_models_characteristics():
    for model_name in models:
        model_path = data_path + f'/{model_name}/vae'
        do_embeddings(preprocessed_data(4), model_path, model_path, get_characteristics_path(model_name),
                      get_emb_path(model_name), 4)
