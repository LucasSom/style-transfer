import keras.models
import pytest

from model.train import train_new_model
from utils.files_utils import load_pickle


@pytest.fixture
def sonata15_mapleleaf_prep():
    return load_pickle(name="sonata15_mapleleaf_ds", path="../data/preprocessed_data/")


def test_new_model_1_epoch(sonata15_mapleleaf_prep):
    epochs = 1
    model, callbacks = train_new_model(sonata15_mapleleaf_prep, "test_new_model", epochs, ckpt=1)
    assert callbacks != {}
    assert set(callbacks.keys()) == {"decoder_loss", "decoder_1_loss", "decoder_2_loss", "decoder_3_loss", "loss"}
    for l in callbacks.values():
        assert len(l) == epochs + 1


def test_new_model_5_epochs(sonata15_mapleleaf_prep):
    epochs = 5
    model, callbacks = train_new_model(sonata15_mapleleaf_prep, "test_new_model", epochs, ckpt=2)
    assert callbacks != {}
    assert set(callbacks.keys()) == {"decoder_loss", "decoder_1_loss", "decoder_2_loss", "decoder_3_loss", "loss"}
    for l in callbacks.values():
        assert len(l) == epochs + 1


def test_new_model_1_epoch_pkl():
    epochs = 1
    model, callbacks = train_new_model("mapleleaf_ds", "test_new_model", epochs, ckpt=1)
    assert callbacks != {}
    assert set(callbacks.keys()) == {"decoder_loss", "decoder_1_loss", "decoder_2_loss", "decoder_3_loss", "loss"}
    for l in callbacks.values():
        assert len(l) == epochs + 1


def test_train_model(sonata15_mapleleaf_prep):
    ...


def test_plot_visualization():
    ...


'''
from model.train import train_model

# rolls_preprocessed = np.stack([r.roll for r in df['roll']])
# vae, callbacks = train_model(rolls_preprocessed, if_train, entrenar_nuevo, epoca_final)
vae, callbacks = train_model(df_midi, if_train, entrenar_nuevo, epoca_final)

plot_train(callbacks, epoca_final)
'''
