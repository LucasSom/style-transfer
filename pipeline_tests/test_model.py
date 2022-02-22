import pytest

from model.train import train_new_model, train_model
from utils.files_utils import load_pickle
from utils.plots_utils import plot_train


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


def test_new_model_1_epoch_pkl(model_name="test_new_model"):
    epochs = 1
    model, callbacks = train_new_model("mapleleaf_ds", model_name, epochs, ckpt=1)
    assert callbacks != {}
    assert set(callbacks.keys()) == {"decoder_loss", "decoder_1_loss", "decoder_2_loss", "decoder_3_loss", "loss"}
    for l in callbacks.values():
        assert len(l) == epochs + 1


def test_train_model():
    epochs = 5
    model, callbacks = train_model("mapleleaf_ds", "test_new_model", epochs, ckpt=2)
    assert callbacks != {}
    assert set(callbacks.keys()) == {"decoder_loss", "decoder_1_loss", "decoder_2_loss", "decoder_3_loss", "loss"}
    for l in callbacks.values():
        assert len(l) == epochs + 1 - 2


def test_complete_training():
    test_new_model_1_epoch_pkl("test_complete_training")
    epoch_after_init = 1

    final_epoch_1 = 2
    model, callbacks = train_model("mapleleaf_ds", "test_complete_training", final_epoch_1, ckpt=1)
    assert callbacks != {}
    assert set(callbacks.keys()) == {"decoder_loss", "decoder_1_loss", "decoder_2_loss", "decoder_3_loss", "loss"}
    for l in callbacks.values():
        assert len(l) == final_epoch_1 - epoch_after_init
        print(f"Después de la primera corrida: {len(l)}")

    final_epoch_2 = 4
    model, callbacks = train_model("mapleleaf_ds", "test_complete_training", final_epoch_2, ckpt=1)
    assert callbacks != {}
    assert set(callbacks.keys()) == {"decoder_loss", "decoder_1_loss", "decoder_2_loss", "decoder_3_loss", "loss"}
    for l in callbacks.values():
        assert len(l) == final_epoch_2 - final_epoch_1
        print(f"Después de la segunda: {len(l)}")
    return callbacks


def test_plot_visualization():
    final_epoch = 1 + 2 + 4
    callbacks = test_complete_training()
    plot_train(callbacks, final_epoch)
