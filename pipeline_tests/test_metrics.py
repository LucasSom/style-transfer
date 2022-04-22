import numpy as np
import pytest

from model.colab_tension_vae.params import init
from utils.files_utils import data_tests_path, load_pickle


@pytest.fixture
def matrix_1bar():
    return np.loadtxt(data_tests_path + "matrix_1bar.csv", delimiter=",", dtype=int)


@pytest.fixture
def roll_8bar_w_rest():
    init("8bar")
    return load_pickle("roll_8bar_w_rest", data_tests_path)


# def test_intervals_list_conversion_from_matrix(matrix_1bar):
#     init('1bar')
#     intervals = GuoRoll(matrix_1bar).get_adjacent_intervals()
#     assert intervals == [1, -1, -2, 7, -4, -1]


def test_intervals_list_conversion_from_roll(roll_8bar_w_rest):

    intervals = roll_8bar_w_rest.get_adjacent_intervals(voice='melody')
    assert intervals == [0, 0, 0, 2, 2, 0, 0, 3, -7, 2, 2, -2, -2, 4, -2, 2, -4, 2, -3, 1, 0, 0, 2, -2, -1, -2]
