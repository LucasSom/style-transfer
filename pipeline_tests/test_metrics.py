import numpy as np
import pytest

from model.colab_tension_vae.params import init
from roll.guoroll import GuoRoll
from utils.files_utils import data_tests_path, load_pickle


@pytest.fixture
def matrix_4bar():
    return np.loadtxt(data_tests_path + "matrix_4bar.csv", delimiter=",", dtype=int)


@pytest.fixture
def roll_8bar_w_rest():
    init("8bar")
    return load_pickle(data_tests_path + "roll_8bar_w_rest")


def pattern_to_str(pattern):
    s = ""
    for i in pattern:
        s += str(int(i))
    return s


# def test_intervals_list_conversion_from_matrix(matrix_1bar):
#     init('1bar')
#     intervals = GuoRoll(matrix_1bar).get_adjacent_intervals()
#     assert intervals == [1, -1, -2, 7, -4, -1]


def test_intervals_list_conversion_from_roll(roll_8bar_w_rest):
    intervals = roll_8bar_w_rest.get_adjacent_intervals(voice='melody')
    assert intervals == [0, 0, 0, 2, 2, 0, 0, 3, -7, 2, 2, -2, -2, 4, -2, 2, -4, 2, -3, 1, 0, 0, 2, -2, -1, -2]


def test_rhythmic_patters_choral_melody(roll_8bar_w_rest):
    r_patterns = roll_8bar_w_rest.get_adjacent_rhythmic_patterns(voice='melody')
    correct_patterns = ['1000', '1000', '1000', '1000',
                        '1000', '1000', '0000', '0000',
                        '1000', '1000', '1000', '1010',
                        '1000', '1000', '0000', '1000',
                        '1000', '1000', '1000', '1000',
                        '1000', '1000', '0000', '0000',
                        '1000', '1000', '1000', '1000',
                        '1000', '1000', '0000', '1000']
    assert len(r_patterns) == len(correct_patterns)
    r_patterns = list(map(pattern_to_str, r_patterns))
    for i, (r, c) in enumerate(zip(r_patterns, correct_patterns)):
        try:
            assert c == r
        except:
            print(f"\n---------------EL GRAN CULPABLE FUE: {i}. Esperábamos {c}---------------")
            assert r == c


def test_rhythmic_patters_choral_bass(roll_8bar_w_rest):
    r_patterns = roll_8bar_w_rest.get_adjacent_rhythmic_patterns(voice='bass')
    correct_patterns = ['1000', '1000', '1000', '1000',
                        '1000', '1000', '0000', '0000',
                        '1010', '1000', '1000', '1000',
                        '1010', '1000', '0000', '1000',
                        '1000', '1010', '1000', '1000',
                        '1000', '1000', '0000', '0000',
                        '1010', '1010', '1010', '1000',
                        '1000', '1000', '0000', '1000']

    assert len(r_patterns) == len(correct_patterns)
    r_patterns = list(map(pattern_to_str, r_patterns))
    for i, (r, c) in enumerate(zip(r_patterns, correct_patterns)):
        try:
            assert c == r
        except:
            print(f"\n---------------EL GRAN CULPABLE FUE: {i}. Esperábamos {c}---------------")
            assert r == c



def test_rhythmic_patters_multiple(matrix_4bar):
    init(4)
    roll_4bar = GuoRoll(matrix_4bar)
    r_patterns = roll_4bar.get_adjacent_rhythmic_patterns(voice='melody')
    correct_patterns = ['qq', 'sqs', 'dr', 'rsq',
                        'sssr', 'srrs', 'ds', 'ssq',
                        'rqr', 'sd', 'qsr', 'rsrr',
                        'rrrr', 'qss', 'qrr', 'rqs']
    assert len(r_patterns) == len(correct_patterns)
    assert r_patterns == correct_patterns
