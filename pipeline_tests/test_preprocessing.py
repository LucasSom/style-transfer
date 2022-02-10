from debug_utils import pm_cmp
from display_audio import save_audios
from files_utils import save_pickle, load_pickle
from preprocessing import preprocess_data
import pytest

from roll.roll import roll_to_midi


@pytest.fixture
def sonata15_mapleleaf_ds():
    return {"mozart_test": ["../data/debug/sonata15-1-debug.mid"],
            "ragtime_test": ["../data/debug/mapleleaf.mid"]}


@pytest.fixture
def mapleleaf_ds():
    return {"ragtime_test": ["../data/debug/mapleleaf.mid"]}


def test_preprocess_midi():
    pass


def test_mapleaf(mapleleaf_ds):
    try:
        df = load_pickle(name="mapleleaf_ds", path="../data/debug/")
    except:
        df = preprocess_data(mapleleaf_ds)
        save_pickle(df, name="mapleleaf_ds", path="../data/debug/")

    save_audios(df.midi[19], path="../data/debug/")
    assert df[df["Autor"] == "ragtime_test"].shape[0] == 17
    assert pm_cmp(df.midi[0], roll_to_midi(df.roll[0], df.oldPM[0]))


def test_preprocess_data(sonata15_mapleleaf_ds):
    try:
        df = load_pickle(name="sonata15_mapleleaf_ds", path="../data/debug/")
    except:
        df = preprocess_data(sonata15_mapleleaf_ds)
        save_pickle(df, name="sonata15_mapleleaf_ds", path="../data/debug/")

    assert df[df["Autor"] == "mozart_test"].shape[0] <= 18
    assert df[df["Autor"] == "mozart_test"].shape[0] > 0
    assert df[df["Autor"] == "ragtime_test"].shape[0] <= 17
    assert df[df["Autor"] == "ragtime_test"].shape[0] > 0
    assert pm_cmp(df.midi[0], roll_to_midi(df.roll[0], df.oldPM[0]))
    assert pm_cmp(df.midi[20], roll_to_midi(df.roll[20], df.oldPM[20]))


def test_midis_from_df(sonata15_mapleleaf_ds):
    try:
        df = load_pickle(name="sonata15_mapleleaf_ds", path="../data/debug/")
    except:
        df = preprocess_data(sonata15_mapleleaf_ds)
        save_pickle(df, name="sonata15_mapleleaf_ds", path="../data/debug/")

    save_audios([(df["Titulo"][0], df["rollID"][0], df["midi"][0], df["oldPM"][0]),
                 (df["Titulo"][20], df["rollID"][20], df["midi"][20], df["oldPM"][20])
                 ],
                path="../data/debug/")
