import os

import model.colab_tension_vae.params as params
from utils.debug_utils import pm_cmp
from utils.audio_management import save_audios
from utils.files_utils import save_pickle, load_pickle, data_path, preprocessed_data_path, datasets_path, \
    original_audios_path
from preprocessing import preprocess_data
import pytest

from roll.guoroll import GuoRoll


@pytest.fixture
def sonata15_mapleleaf_ds():
    return {"mozart_test": [os.path.join(datasets_path, "debug/sonata15-1-debug.mid")],
            "ragtime_test": [os.path.join(datasets_path, "debug/mapleleaf.mid")]}


@pytest.fixture
def mapleleaf_ds():
    return {"ragtime_test": [os.path.join(datasets_path, "debug/mapleleaf.mid")]}


# @pytest.fixture
# def bach_ds():
#     return {"bach": [data_path + "Bach/" + path for path in os.listdir(data_path + "Bach/")]}


# @pytest.fixture
# def breeze_ds():
#     return {"breeze_test": [data_path + "ragtime/breeze.mid"]}


# def test_breeze_preprocessing(breeze_ds):
#     params.init()
#     preprocess_data(breeze_ds)
#     assert True


# def test_not_cached(sonata15_mapleleaf_ds):
#     assert os.system(f"python3 show_statistics.py {data_path}") == 0


def test_mapleaf(mapleleaf_ds):
    params.init("8bar")
    try:
        df = load_pickle(file_name=preprocessed_data_path + "mapleleaf_ds-8")
    except:
        df = preprocess_data(mapleleaf_ds)
        save_pickle(df, file_name=preprocessed_data_path + "mapleleaf_ds-8")

    audio_path = os.path.join(data_path, "debug_outputs/audios/")

    roll = df["roll"][0]
    save_audios([df["Titulo"][0]], [roll.midi], path=audio_path)
    assert df[df["Autor"] == "ragtime_test"].shape[0] <= 17
    assert df[df["Autor"] == "ragtime_test"].shape[0] > 0
    r = GuoRoll(df.roll[0].matrix, 'mapleleaf_8')
    assert pm_cmp(r.midi, r._roll_to_midi(old_pm=df.roll[0].song.old_pm, path=audio_path))


def test_preprocess_data(sonata15_mapleleaf_ds):
    params.init("8bar")
    try:
        df = load_pickle(file_name=preprocessed_data_path + "sonata15_mapleleaf_ds-8")
    except:
        df = preprocess_data(sonata15_mapleleaf_ds)
        save_pickle(df, file_name=preprocessed_data_path + "sonata15_mapleleaf_ds-8")

    audio_path = os.path.join(data_path, "debug_outputs/audios/")

    assert df[df["Autor"] == "mozart_test"].shape[0] <= 18
    assert df[df["Autor"] == "mozart_test"].shape[0] > 0
    assert df[df["Autor"] == "ragtime_test"].shape[0] <= 17
    assert df[df["Autor"] == "ragtime_test"].shape[0] > 0

    r0 = GuoRoll(df.roll[0].matrix, 'matrix_test_0')
    r20 = GuoRoll(df.roll[20].matrix, 'matrix_test_20')
    assert pm_cmp(r0.midi, r0._roll_to_midi(old_pm=df.roll[0].song.old_pm, path=audio_path))
    assert pm_cmp(r20.midi, r20._roll_to_midi(old_pm=df.roll[20].song.old_pm, path=audio_path))


def test_midis_from_df(sonata15_mapleleaf_ds):
    params.init("8bar")
    try:
        df = load_pickle(file_name=preprocessed_data_path + "sonata15_mapleleaf_ds-8")
    except:
        df = preprocess_data(sonata15_mapleleaf_ds)
        save_pickle(df, file_name=preprocessed_data_path + "sonata15_mapleleaf_ds-8")
    r0 = df["roll"][0]
    r20 = df["roll"][20]
    save_audios([df["Titulo"][0], df["Titulo"][20]], [r0.midi, r20.midi], path=data_path + "/debug_outputs/audios/")


# def test_preprocess_bach(bach_ds):
#     params.init("8bar")
#     preprocess_data(bach_ds)
#     assert True


# def test_breeze_preprocessing_4bars(breeze_ds):
#     params.init("8bar")
#     df = preprocess_data(breeze_ds)
#     assert df['roll'][0].matrix.shape == (df['roll'][0].bars * params.config.SAMPLES_PER_BAR, 89)


def test_mapleaf_4bars(mapleleaf_ds):
    params.init("4bar")
    try:
        df = load_pickle(file_name=preprocessed_data_path + "mapleleaf_ds-4")
    except:
        df = preprocess_data(mapleleaf_ds)
        save_pickle(df, file_name=preprocessed_data_path + "mapleleaf_ds-4")

    # audio_path = os.path.join(data_path, "debug_outputs/audios/")
    audio_path = original_audios_path

    roll = df["roll"][0]
    save_audios([df["Titulo"][0]], [roll.midi], path=audio_path)
    assert df[df["Autor"] == "ragtime_test"].shape[0] <= 17 * 2
    assert df[df["Autor"] == "ragtime_test"].shape[0] > 0
    r = GuoRoll(df.roll[0].matrix, 'mapleleaf_4')
    assert pm_cmp(r.midi, r._roll_to_midi(old_pm=df.roll[0].song.old_pm, path=audio_path))
    assert df['roll'][0].matrix.shape == (df['roll'][0].bars * params.config.SAMPLES_PER_BAR, 89)


def test_preprocess_data_4bars(sonata15_mapleleaf_ds):
    params.init("4bar")
    try:
        df = load_pickle(file_name=preprocessed_data_path + "sonata15_mapleleaf_ds-4")
    except:
        df = preprocess_data(sonata15_mapleleaf_ds)
        save_pickle(df, file_name=preprocessed_data_path + "sonata15_mapleleaf_ds-4")

    audio_path = os.path.join(data_path, "debug_outputs/audios/")

    r0 = GuoRoll(df.roll[0].matrix, 'matrix_test_0')
    r20 = GuoRoll(df.roll[20].matrix, 'matrix_test_20')
    assert pm_cmp(r0.midi, r0._roll_to_midi(old_pm=df.roll[0].song.old_pm, path=audio_path))
    assert pm_cmp(r20.midi, r20._roll_to_midi(old_pm=df.roll[20].song.old_pm, path=audio_path))
    assert df['roll'][0].matrix.shape == (df['roll'][0].bars * params.config.SAMPLES_PER_BAR, 89)


def test_midis_from_df_4bars(sonata15_mapleleaf_ds):
    params.init("4bar")
    try:
        df = load_pickle(file_name=preprocessed_data_path + "sonata15_mapleleaf_ds-4")
    except:
        df = preprocess_data(sonata15_mapleleaf_ds)
        save_pickle(df, file_name=preprocessed_data_path + "sonata15_mapleleaf_ds-4")

    audio_path = os.path.join(data_path, "debug_outputs/audios/")

    r0 = df["roll"][0]
    r20 = df["roll"][20]
    save_audios([df["Titulo"][0], df["Titulo"][20]], [r0.midi, r20.midi], path=audio_path)


# def test_preprocess_bach_4bars(bach_ds):
#     params.init("4bar")
#     preprocess_data(bach_ds)
#     assert True
