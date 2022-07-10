import pandas as pd
import pytest

from model.colab_tension_vae.params import init
from model.train import train_new_model, continue_training
from utils.files_utils import load_pickle, data_path, preprocessed_data_path
from utils.plots_utils import plot_train

cols = ['decoder_loss', 'decoder_1_loss', 'decoder_2_loss', 'decoder_3_loss', 'loss', 'epoch']


@pytest.fixture
def sonata15_mapleleaf_prep_8():
    return load_pickle(file_name=preprocessed_data_path+"sonata15_mapleleaf_ds-8")


@pytest.fixture
def sonata15_mapleleaf_prep_4():
    return load_pickle(file_name=preprocessed_data_path+"sonata15_mapleleaf_ds-4")


def test_new_model_1_epoch(sonata15_mapleleaf_prep_8):
    init("8bar")
    epochs = 1
    model_name = "test_new_model"
    train_new_model(sonata15_mapleleaf_prep_8, model_name, epochs, ckpt=1)
    callbacks = pd.read_csv(data_path + f"logs/{model_name}_0.csv")

    for c in cols:
        assert c in set(callbacks.columns)
    # assert callbacks["loss"].shape[0] == epochs


def test_new_model_5_epochs(sonata15_mapleleaf_prep_8):
    init("8bar")
    epochs = 5
    model_name = "test_new_model"
    train_new_model(sonata15_mapleleaf_prep_8, model_name, epochs, ckpt=2)
    callbacks = pd.read_csv(data_path + f"logs/{model_name}_0.csv")

    for c in cols:
        assert c in set(callbacks.columns)
    # assert callbacks["loss"].shape[0] == epochs


def test_new_model_1_epoch_pkl(sonata15_mapleleaf_prep_8, model_name="test_new_model"):
    init("8bar")
    epochs = 1
    train_new_model(sonata15_mapleleaf_prep_8, model_name, epochs, ckpt=1)
    callbacks = pd.read_csv(data_path + f"logs/{model_name}_0.csv")

    for c in cols:
        assert c in set(callbacks.columns)
    # assert callbacks["loss"].shape[0] == epochs


def test_train_model(sonata15_mapleleaf_prep_8):
    init("8bar")
    epochs = 5
    model_name = "test_new_model"
    continue_training(sonata15_mapleleaf_prep_8, model_name, epochs, ckpt=2)
    callbacks = pd.read_csv(data_path + f"logs/{model_name}_2.csv")

    for c in cols:
        assert c in set(callbacks.columns)
    # assert callbacks["loss"].shape[0] == epochs


def test_complete_training(sonata15_mapleleaf_prep_8):
    model_name = "test_complete_training"
    test_new_model_1_epoch_pkl(sonata15_mapleleaf_prep_8, model_name)

    final_epoch_1 = 2
    continue_training(sonata15_mapleleaf_prep_8, model_name, final_epoch_1, ckpt=1)
    callbacks = pd.read_csv(data_path + f"logs/{model_name}_2.csv")

    for c in cols:
        assert c in set(callbacks.columns)
    # assert callbacks["loss"].shape[0] == final_epoch_1

    final_epoch_2 = 4
    continue_training(sonata15_mapleleaf_prep_8, model_name, final_epoch_2, ckpt=1)
    callbacks = pd.read_csv(data_path + f"logs/{model_name}_5.csv")

    for c in cols:
        assert c in set(callbacks.columns)
    # assert callbacks["loss"].shape[0] == final_epoch_2
    return callbacks


def test_plot_visualization(sonata15_mapleleaf_prep_8):
    init("8bar")
    final_epoch = 5
    model_name = "test_complete_training"
    callbacks = pd.read_csv(data_path + f"logs/{model_name}_5.csv")
    plot_train(callbacks, final_epoch)
    assert True


def test_new_model_1_epoch_4bars(sonata15_mapleleaf_prep_4):
    init("4bar")
    epochs = 1
    model_name = "test_new_model-4"
    train_new_model(sonata15_mapleleaf_prep_4, model_name, epochs, ckpt=1)
    callbacks = pd.read_csv(data_path + f"logs/{model_name}_0.csv")

    for c in cols:
        assert c in set(callbacks.columns)
    # assert callbacks["loss"].shape[0] == epochs


def test_new_model_5_epochs_4bars(sonata15_mapleleaf_prep_4):
    init("4bar")
    epochs = 5
    model_name = "test_new_model-4"
    train_new_model(sonata15_mapleleaf_prep_4, model_name, epochs, ckpt=2)
    callbacks = pd.read_csv(data_path + f"logs/{model_name}_0.csv")

    for c in cols:
        assert c in set(callbacks.columns)
    # assert callbacks["loss"].shape[0] == epochs


def test_new_model_1_epoch_pkl_4bars(sonata15_mapleleaf_prep_4, model_name="test_new_model-4"):
    init("4bar")
    epochs = 1
    train_new_model(sonata15_mapleleaf_prep_4, model_name, epochs, ckpt=1)
    callbacks = pd.read_csv(data_path + f"logs/{model_name}_0.csv")

    for c in cols:
        assert c in set(callbacks.columns)
    # assert callbacks["loss"].shape[0] == epochs


def test_train_model_4bars(sonata15_mapleleaf_prep_4):
    init("4bar")
    epochs = 5
    model_name = "test_new_model-4"
    continue_training(sonata15_mapleleaf_prep_4, model_name, epochs, ckpt=2)
    callbacks = pd.read_csv(data_path + f"logs/{model_name}_2.csv")

    for c in cols:
        assert c in set(callbacks.columns)
    # assert callbacks["loss"].shape[0] == epochs


def test_complete_training_4bars(sonata15_mapleleaf_prep_4):
    model_name = "test_complete_training-4"
    test_new_model_1_epoch_pkl_4bars(sonata15_mapleleaf_prep_4, model_name)

    final_epoch_1 = 2
    continue_training(sonata15_mapleleaf_prep_4, model_name, final_epoch_1, ckpt=1)
    callbacks = pd.read_csv(data_path + f"logs/{model_name}_2.csv")

    for c in cols:
        assert c in set(callbacks.columns)
    # assert callbacks["loss"].shape[0] == final_epoch_1

    final_epoch_2 = 4
    continue_training(sonata15_mapleleaf_prep_4, model_name, final_epoch_2, ckpt=1)
    try:
        callbacks = pd.read_csv(data_path + f"logs/{model_name}_4.csv")
    except:
        callbacks = pd.read_csv(data_path + f"logs/{model_name}_6.csv")

    for c in cols:
        assert c in set(callbacks.columns)
    # assert callbacks["loss"].shape[0] == final_epoch_2
    return callbacks


def test_plot_visualization_4bars(sonata15_mapleleaf_prep_4):
    init("4bar")
    final_epoch = 5
    model_name = "test_complete_training"
    callbacks = pd.read_csv(data_path + f"logs/{model_name}_5.csv")
    plot_train(callbacks, final_epoch)
    assert True
