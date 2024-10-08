import numpy as np
import pandas as pd
import pytest
from tensorflow import keras

from dodo import analyze_training, do_embeddings, models, do_reconstructions
from model.colab_tension_vae.params import init
from model.embeddings.embeddings import obtain_embeddings
from model.embeddings.style import Style
from utils.files_utils import load_pickle, preprocessed_data_dir, get_embedding_path, \
    get_reconstruction_path, get_characteristics_path, get_emb_path, data_path, get_model_paths, preprocessed_data_path
from utils.plots_utils import plot_characteristics_distributions


@pytest.fixture
def brmf4_prep():
    return load_pickle(file_name=preprocessed_data_dir + "bach-rag-moz-fres-4")


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

    df = load_pickle(f'{data_path}models/4-small_br/embeddings/df_emb.pkl')
    df["Style"][:2] = "Pepe"
    df["Style"][2:4] = "Jose"
    df["Style"][4:] = "Baile"

    styles = {
        "Pepe": Style("Pepe", df_char=df_char, df=df),
        "Jose": Style("Jose", df_char=df_char, df=df),
        "Baile": Style("Baile", df_char=df_char, df=df)
    }

    styles["Pepe"].intervals_distribution = np.ones((24, 24))
    styles["Jose"].intervals_distribution = np.ones((24, 24))
    styles["Jose"].intervals_distribution[10, 11] += 1

    styles["Baile"].intervals_distribution = np.zeros((24, 24))
    styles["Baile"].intervals_distribution[10:13, 10:13] += 2

    return styles


def test_obtain_embeddings(brmf4_prep):
    vae = keras.models.load_model(get_model_paths("brmf_4b")[1])

    df_emb = obtain_embeddings(brmf4_prep, vae)

    assert "Embedding" in df_emb.columns
    for e in df_emb.Embedding:
        assert e.shape == (96,)


def test_analyze_training():
    b, z = 4, 96
    model_name = '4-CPFRAa-96'
    _, vae_dir, vae_path = get_model_paths(model_name)
    train_path = f"{preprocessed_data_dir}{model_name}train.pkl"
    analyze_training(train_path=train_path, vae_dir=vae_dir,
                     model_name=model_name, b=4, targets=[get_reconstruction_path(model_name)], z=z)


def test_analyze_training_mixture_model():
    b, z = 4, 96
    model_name = '4-Lakh_Kern-96'
    model_name_aux = f"{b}-CPFRAa-{z}"
    _, vae_dir, vae_path = get_model_paths(model_name_aux)
    analyze_training(train_path=preprocessed_data_path(4, False), vae_dir=vae_dir,
                     model_name=model_name, b=4, targets=[get_reconstruction_path(model_name)], z=z)


def test_reconstruction():
    b, z = 4, 96
    model_name = '4-CPFRAa-96'
    _, vae_dir, vae_path = get_model_paths(model_name)
    emb_path = get_emb_path(model_name)

    do_reconstructions(emb_path, model_name, vae_dir, b, z, [get_reconstruction_path(model_name)])


def test_reconstruction_mixture_model():
    b, z = 4, 96
    model_name = "4-Lakh_Kern-96"
    model_name_aux = f"{b}-CPFRAa-{z}"
    _, vae_dir, vae_path = get_model_paths(model_name_aux)

    emb_path = get_emb_path(model_name)

    do_reconstructions(emb_path, model_name, vae_dir, b, z, [get_reconstruction_path(model_name)])


# def test_plot_distributions(characteristics):
#     plot_dir = data_path + 'debug_outputs/'
#     plot_name = "distributions"
#     plot_characteristics_distributions(characteristics, plot_dir, plot_name)


def test_characteristics():
    model_name = '4-small_br'
    model_path = data_path + f'models/{model_name}/vae'
    do_embeddings(preprocessed_data_path(4, False), val_path, model_path, model_path,
                  get_characteristics_path(model_name), get_emb_path(model_name), 4, z=96)


def test_characteristics_beta():
    model_name = 'brmf_4b_beta'
    b, z = 4, 96

    model_path, vae_dir, vae_path = get_model_paths(model_name)
    characteristics_path = get_characteristics_path(model_name)
    emb_path = get_emb_path(model_name)
    small = "small" in model_name

    do_embeddings(preprocessed_data_path(b, False, small), val_path, model_path, vae_dir, characteristics_path,
                  emb_path, b, z)

# def test_all_models_characteristics():
#     for model_name in models:
#         model_path = data_path + f'models/{model_name}/vae'
#         do_embeddings(preprocessed_data_path(4, False), model_path, model_path, get_characteristics_path(model_name),
#                       get_emb_path(model_name), 4, z=96)
