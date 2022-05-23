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
    return load_pickle(data_tests_path + "roll_8bar_w_rest")


# def test_intervals_list_conversion_from_matrix(matrix_1bar):
#     init('1bar')
#     intervals = GuoRoll(matrix_1bar).get_adjacent_intervals()
#     assert intervals == [1, -1, -2, 7, -4, -1]


def test_intervals_list_conversion_from_roll(roll_8bar_w_rest):
    intervals = roll_8bar_w_rest.get_adjacent_intervals(voice='melody')
    assert intervals == [0, 0, 0, 2, 2, 0, 0, 3, -7, 2, 2, -2, -2, 4, -2, 2, -4, 2, -3, 1, 0, 0, 2, -2, -1, -2]


def test_rhythmic_patters(roll_8bar_w_rest):
    r_patterns = roll_8bar_w_rest.get_adjacent_rhythmic_patterns(voice='melody')
    correct_patterns = ['c', 'c', 'c', 'c',
                          'c', 'c', 'c', 'rrrr',
                          'c', 'c', 'c', 'qq',
                          'c', 'c', 'c', 'c',
                          'c', 'c', 'c', 'c',
                          'c', 'c', 'c', 'rrrr',
                          'c', 'c', 'c', 'c',
                          'c', 'c', 'c', 'c']
    assert len(r_patterns) == len(correct_patterns)
    assert r_patterns == correct_patterns
