import os

from utils.debug_utils import pm_cmp
from display_audio import save_audios
from utils.files_utils import save_pickle, load_pickle
from preprocessing import preprocess_data
import pytest

from roll.guoroll import GuoRoll


@pytest.fixture
def sonata15_mapleleaf_ds():
    return {"mozart_test": ["../data/debug/sonata15-1-debug.mid"],
            "ragtime_test": ["../data/debug/mapleleaf.mid"]}


@pytest.fixture
def mapleleaf_ds():
    return {"ragtime_test": ["../data/debug/mapleleaf.mid"]}


def test_not_cached(sonata15_mapleleaf_ds):
    assert os.system("python3 show_statistics.py /home/urania/Documentos/Tesis/src/style-transfer/data/") == 0


def test_mapleaf(mapleleaf_ds):
    try:
        df = load_pickle(name="mapleleaf_ds", path="../data/preprocessed_data/")
    except:
        df = preprocess_data(mapleleaf_ds)
        save_pickle(df, name="mapleleaf_ds", path="../data/preprocessed_data/")

    roll = df["roll"][0]
    save_audios([(df["Titulo"][0], roll.midi, roll.song.old_pm)], path="../data/debug_outputs/audios/")
    assert df[df["Autor"] == "ragtime_test"].shape[0] <= 17
    assert df[df["Autor"] == "ragtime_test"].shape[0] > 0
    r = GuoRoll(df.roll[0].matrix, bars=8)
    assert pm_cmp(r.midi, r._roll_to_midi(df.roll[0].song.old_pm))


def test_preprocess_data(sonata15_mapleleaf_ds):
    try:
        df = load_pickle(name="sonata15_mapleleaf_ds", path="../data/preprocessed_data/")
    except:
        df = preprocess_data(sonata15_mapleleaf_ds)
        save_pickle(df, name="sonata15_mapleleaf_ds", path="../data/preprocessed_data/")

    assert df[df["Autor"] == "mozart_test"].shape[0] <= 18
    assert df[df["Autor"] == "mozart_test"].shape[0] > 0
    assert df[df["Autor"] == "ragtime_test"].shape[0] <= 17
    assert df[df["Autor"] == "ragtime_test"].shape[0] > 0

    r0 = GuoRoll(df.roll[0].matrix, bars=8)
    r20 = GuoRoll(df.roll[20].matrix, bars=8)
    assert pm_cmp(r0.midi, r0._roll_to_midi(df.roll[0].song.old_pm))
    assert pm_cmp(r20.midi, r20._roll_to_midi(df.roll[20].song.old_pm))


def test_midis_from_df(sonata15_mapleleaf_ds):
    try:
        df = load_pickle(name="sonata15_mapleleaf_ds", path="../data/preprocessed_data/")
    except:
        df = preprocess_data(sonata15_mapleleaf_ds)
        save_pickle(df, name="sonata15_mapleleaf_ds", path="../data/preprocessed_data/")
    r0 = df["roll"][0]
    r20 = df["roll"][20]
    save_audios([(df["Titulo"][0], r0.midi, r0.song.old_pm),
                 (df["Titulo"][20], r20.midi, r20.song.old_pm)
                 ],
                path="../data/debug_outputs/audios/")
