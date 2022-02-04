from debug_utils import pm_cmp
from files_utils import save_pickle, load_pickle
from preprocessing import preprocess_data
import pytest

from roll.roll import roll_to_midi


@pytest.fixture
def sonata15_antoinette_ds():
    return {"mozart_test": ["../data/debug/sonata15-1-debug.mid"],
            "ragtime_test": ["../data/debug/antoinette.mid"]}


def test_preprocess_data(sonata15_antoinette_ds):
    try:
        df = load_pickle("../data/debug/sonata15_antoinette_ds.pkl")
    except:
        df = preprocess_data(sonata15_antoinette_ds)
        save_pickle(df, name="sonata15_antoinette_ds.pkl", path="../data/debug/")

    assert df[df["Autor"] == "mozart_test"].shape[0] == 19
    assert pm_cmp(df.midi[0], roll_to_midi(df.roll[0], df.oldPM[0]))

