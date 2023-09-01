import os

import model.colab_tension_vae.params as params
from dodo import preprocess, subdataset_lmd
from utils.debug_utils import pm_cmp
from utils.audio_management import save_audios
from utils.files_utils import save_pickle, load_pickle, data_path, preprocessed_data_dir, datasets_path, \
    original_audios_path, preprocessed_data_path
from preprocessing.preprocessing import preprocess_data
import pytest

from roll.guoroll import GuoRoll


@pytest.fixture
def sonata15_mapleleaf_ds():
    return {"mozart_test": [os.path.join(datasets_path, "debug/sonata15-1-debug.mid")],
            "ragtime_test": [os.path.join(datasets_path, "debug/mapleleaf.mid")]}


@pytest.fixture
def mapleleaf_ds():
    return {"ragtime_test": [os.path.join(datasets_path, "debug/mapleleaf.mid")]}


def test_mapleaf(mapleleaf_ds):
    params.init(8)
    try:
        df = load_pickle(file_name=preprocessed_data_dir + "mapleleaf_ds-8")
    except:
        df = preprocess_data(mapleleaf_ds, False, False)
        save_pickle(df, file_name=preprocessed_data_dir + "mapleleaf_ds-8")

    audio_path = os.path.join(data_path, "debug_outputs/audios/")

    roll = df["roll"][0]
    save_audios([df["Title"][0]], [roll.midi], path=audio_path)
    assert df[df["Style"] == "ragtime_test"].shape[0] <= 17
    assert df[df["Style"] == "ragtime_test"].shape[0] > 0
    r = GuoRoll(df.roll[0].matrix, 'mapleleaf_8')
    assert pm_cmp(r.midi, r.roll_to_audio(path=audio_path, old_pm=df.roll[0].song.old_pm))


def test_preprocess_data(sonata15_mapleleaf_ds):
    params.init(8)
    try:
        df = load_pickle(file_name=preprocessed_data_dir + "sonata15_mapleleaf_ds-8")
    except:
        df = preprocess_data(sonata15_mapleleaf_ds, False, False)
        save_pickle(df, file_name=preprocessed_data_dir + "sonata15_mapleleaf_ds-8")

    audio_path = os.path.join(data_path, "debug_outputs/audios/")

    assert df[df["Style"] == "mozart_test"].shape[0] <= 18
    assert df[df["Style"] == "mozart_test"].shape[0] > 0
    assert df[df["Style"] == "ragtime_test"].shape[0] <= 17
    assert df[df["Style"] == "ragtime_test"].shape[0] > 0

    r0 = GuoRoll(df.roll[0].matrix, 'matrix_test_0')
    r20 = GuoRoll(df.roll[20].matrix, 'matrix_test_20')
    assert pm_cmp(r0.midi, r0.roll_to_audio(path=audio_path, old_pm=df.roll[0].song.old_pm))
    assert pm_cmp(r20.midi, r20.roll_to_audio(path=audio_path, old_pm=df.roll[20].song.old_pm))


def test_midis_from_df(sonata15_mapleleaf_ds):
    params.init(8)
    try:
        df = load_pickle(file_name=preprocessed_data_dir + "sonata15_mapleleaf_ds-8")
    except:
        df = preprocess_data(sonata15_mapleleaf_ds, False, False)
        save_pickle(df, file_name=preprocessed_data_dir + "sonata15_mapleleaf_ds-8")
    r0 = df["roll"][0]
    r20 = df["roll"][20]
    save_audios([df["Title"][0], df["Title"][20]], [r0.midi, r20.midi], path=data_path + "/debug_outputs/audios/")


def test_mapleaf_4bars(mapleleaf_ds):
    params.init(4)
    try:
        df = load_pickle(file_name=preprocessed_data_dir + "mapleleaf_ds-4")
    except:
        df = preprocess_data(mapleleaf_ds, False, False)
        save_pickle(df, file_name=preprocessed_data_dir + "mapleleaf_ds-4")

    audio_path = original_audios_path

    roll = df["roll"][0]
    save_audios([df["Title"][0]], [roll.midi], path=audio_path)
    assert df[df["Style"] == "ragtime_test"].shape[0] <= 17 * 2
    assert df[df["Style"] == "ragtime_test"].shape[0] > 0
    r = GuoRoll(df.roll[0].matrix, 'mapleleaf_4')
    assert pm_cmp(r.midi, r.roll_to_audio(path=audio_path, old_pm=df.roll[0].song.old_pm))
    assert df['roll'][0].matrix.shape == (df['roll'][0].bars * params.config.SAMPLES_PER_BAR, 89)


def test_preprocess_data_4bars(sonata15_mapleleaf_ds):
    params.init(4)
    try:
        df = load_pickle(file_name=preprocessed_data_dir + "sonata15_mapleleaf_ds-4")
    except:
        df = preprocess_data(sonata15_mapleleaf_ds, False, False)
        save_pickle(df, file_name=preprocessed_data_dir + "sonata15_mapleleaf_ds-4")

    audio_path = os.path.join(data_path, "debug_outputs/audios/")

    r0 = GuoRoll(df.roll[0].matrix, 'matrix_test_0')
    r20 = GuoRoll(df.roll[20].matrix, 'matrix_test_20')
    assert pm_cmp(r0.midi, r0.roll_to_audio(path=audio_path, old_pm=df.roll[0].song.old_pm))
    assert pm_cmp(r20.midi, r20.roll_to_audio(path=audio_path, old_pm=df.roll[20].song.old_pm))
    assert df['roll'][0].matrix.shape == (df['roll'][0].bars * params.config.SAMPLES_PER_BAR, 89)


def test_midis_from_df_4bars(sonata15_mapleleaf_ds):
    params.init(4)
    try:
        df = load_pickle(file_name=preprocessed_data_dir + "sonata15_mapleleaf_ds-4")
    except:
        df = preprocess_data(sonata15_mapleleaf_ds, False, False)
        save_pickle(df, file_name=preprocessed_data_dir + "sonata15_mapleleaf_ds-4")

    audio_path = os.path.join(data_path, "debug_outputs/audios/")

    r0 = df["roll"][0]
    r20 = df["roll"][20]
    save_audios([df["Title"][0], df["Title"][20]], [r0.midi, r20.midi], path=audio_path)


def test_task_preprocess():
    b = 4
    i = 0
    params.init(b)
    targets = [preprocessed_data_path(b, i+1)]
    folders = [f'{subdataset_lmd}/{i}']
    preprocess(b, folders, save_midis=False, sparse=True, targets=targets)
