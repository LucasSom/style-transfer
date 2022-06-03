import numpy as np
import pytest

from evaluation.metrics.intervals import plot_matrix_of_adjacent_intervals
from evaluation.metrics.rhythmic_patterns import plot_matrix_of_adjacent_rhythmic_patterns
from model.colab_tension_vae.params import init
from roll.guoroll import GuoRoll, pattern_to_str
from roll.song import Song
from utils.files_utils import data_tests_path, load_pickle, data_path


@pytest.fixture
def matrix_4bar():
    return np.loadtxt(data_tests_path + "matrix_4bar.csv", delimiter=",", dtype=int)


@pytest.fixture
def roll_8bar_w_rest():
    init("8bar")
    return load_pickle(data_tests_path + "roll_8bar_w_rest")


def test_intervals_list_conversion_from_roll(roll_8bar_w_rest):
    intervals = roll_8bar_w_rest.get_adjacent_intervals(voice='melody')
    assert intervals == [0, 0, 0, 2, 2, 0, 0, 3, -7, 2, 2, -2, -2, 4, -2, 2, -4, 2, -3, 1, 0, 0, 2, -2, -1, -2]


def assert_patterns(correct_patterns, r_patterns):
    for i, (r, c) in enumerate(zip(r_patterns, correct_patterns)):
        try:
            assert c == r
        except:
            print(f"\n---------------EL GRAN CULPABLE FUE: {i}. Esperábamos {c}---------------")
            assert r == c


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
    assert_patterns(correct_patterns, r_patterns)


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
    assert_patterns(correct_patterns, r_patterns)


def test_rhythmic_patterns_multiple(matrix_4bar):
    init(4)
    roll_4bar = GuoRoll(matrix_4bar)
    r_patterns = roll_4bar.get_adjacent_rhythmic_patterns(voice='melody')
    correct_patterns = ['1010', '1101', '0000', '0110',
                        '1110', '1001', '0001', '0110',
                        '0100', '1100', '0010', '0100',
                        '0000', '1011', '0000', '0101']
    assert len(r_patterns) == len(correct_patterns)
    assert_patterns(correct_patterns, r_patterns)


def test_plot_interval_matrix():
    init(4)
    s = Song(midi_file=f"{data_path}Mozart/sonata15-1-debug.mid", nombre="sonata15")
    plot_matrix_of_adjacent_intervals(s, 'melody')
    plot_matrix_of_adjacent_intervals(s, 'bass')


def test_plot_rhythmic_matrix():
    init(4)
    s = Song(midi_file=f"{data_path}Mozart/sonata15-1-debug.mid", nombre="sonata15")
    plot_matrix_of_adjacent_rhythmic_patterns(s, 'melody')
    plot_matrix_of_adjacent_rhythmic_patterns(s, 'bass')
