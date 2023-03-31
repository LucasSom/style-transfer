import numpy as np
import pytest
from matplotlib import pyplot as plt

from dodo import styles_names, calculate_metrics
from evaluation.metrics.intervals import plot_matrix_of_adjacent_intervals
from evaluation.metrics.metrics import obtain_metrics
from evaluation.metrics.plagiarism import dumb_pitch_plagiarism
from evaluation.metrics.rhythmic_bigrams import plot_matrix_of_adjacent_rhythmic_bigrams, pattern_to_int
from model.colab_tension_vae.params import init
from roll.guoroll import GuoRoll
from roll.song import Song
from utils.files_utils import data_tests_path, load_pickle, original_audios_path, datasets_debug_path, \
    get_transferred_path, get_metrics_dir, get_characteristics_path


@pytest.fixture
def matrix_4bar():
    return np.loadtxt(data_tests_path + "matrix_4bar.csv", delimiter=",", dtype=int)


@pytest.fixture
def matrix_4bar_diff():
    return np.loadtxt(data_tests_path + "matrix_4bar_diff.csv", delimiter=",", dtype=int)


@pytest.fixture
def matrix_4bar_rest_diff():
    return np.loadtxt(data_tests_path + "matrix_4bar_rest_diff.csv", delimiter=",", dtype=int)


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
    roll_4bar = GuoRoll(matrix_4bar, 'matrix_4bar')
    r_patterns = roll_4bar.get_adjacent_rhythmic_patterns(voice='melody')
    correct_patterns = ['1010', '1101', '0000', '0110',
                        '1110', '1001', '0001', '0110',
                        '0100', '1100', '0010', '0100',
                        '0000', '1011', '0000', '0101']
    assert len(r_patterns) == len(correct_patterns)
    assert_patterns(correct_patterns, r_patterns)


def test_plot_interval_matrix():
    init(4)
    s = Song(midi_file=f"{datasets_debug_path}/sonata15-1-debug.mid", nombre="sonata15",
             audio_path=original_audios_path)
    plot_matrix_of_adjacent_intervals(s, 'melody')
    plt.show()
    plot_matrix_of_adjacent_intervals(s, 'bass')
    plt.show()


def test_pattern_to_int():
    assert pattern_to_int('0000') == 0
    assert pattern_to_int('0001') == 1
    assert pattern_to_int('0010') == 2
    assert pattern_to_int('1000') == 8
    assert pattern_to_int('1001') == 9
    assert pattern_to_int('0110') == 6
    assert pattern_to_int('1111') == 15


def test_plot_rhythmic_matrix():
    init(4)
    s = Song(midi_file=f"{datasets_debug_path}/sonata15-1-debug.mid", nombre="sonata15",
             audio_path=original_audios_path)
    plot_matrix_of_adjacent_rhythmic_bigrams(s, 'melody')
    plt.show()
    plot_matrix_of_adjacent_rhythmic_bigrams(s, 'bass')
    plt.show()


def test_dumb_plagiarism_0():
    init(4)
    s = Song(midi_file=f"{datasets_debug_path}/sonata15-1-debug.mid", nombre="sonata15",
             audio_path=original_audios_path)

    b, m = dumb_pitch_plagiarism(s.rolls[0], s.rolls[0])
    assert b, m == (0, 0)


def test_dumb_plagiarism_little_diffs(matrix_4bar, matrix_4bar_diff):
    init(4)
    roll_4bar = GuoRoll(matrix_4bar, 'matrix_4bar')
    roll_4bar_diff = GuoRoll(matrix_4bar_diff, 'matrix_4bar_diff')

    b, m = dumb_pitch_plagiarism(roll_4bar, roll_4bar_diff)
    assert b[1], m[1] == (3, 2)


def test_dumb_plagiarism_rest_diffs(matrix_4bar, matrix_4bar_diff, matrix_4bar_rest_diff):
    init(4)
    roll_4bar = GuoRoll(matrix_4bar, 'matrix_4bar')
    roll_4bar_diff = GuoRoll(matrix_4bar_diff, 'matrix_4bar_diff')
    roll_4bar_rest_diff = GuoRoll(matrix_4bar_rest_diff, 'matrix_4bar_rest_diff')

    b, m = dumb_pitch_plagiarism(roll_4bar, roll_4bar_rest_diff)
    assert b, m == (12, 12)

    b, m = dumb_pitch_plagiarism(roll_4bar, roll_4bar_rest_diff, rest_value=100)
    assert b, m == (100, 100)

    b, m = dumb_pitch_plagiarism(roll_4bar_diff, roll_4bar_rest_diff)
    assert b, m == (15, 14)


def test_rhythm_dumb_plagiarism():
    pass



# -------------------------------- TASK --------------------------------
def test_obtain_metrics_intervals():
    init(4)
    model_name = "brmf_4b"
    e_orig, e_dest = "Bach", "Mozart"

    df = load_pickle(get_transferred_path(e_orig, e_dest, model_name))

    # obtain_metrics(df, e_orig, e_dest, styles, 'rhythmic_bigrams', 'plagiarism', 'intervals')
    d = obtain_metrics(df, e_orig, e_dest, 'intervals')
    print("")
    print(d["intervals"])


def test_obtain_metrics_plagiarism():
    init(4)
    model_name = "brmf_4b"
    e_orig, e_dest = "Bach", "Mozart"

    df = load_pickle(get_transferred_path(e_orig, e_dest, model_name))

    d = obtain_metrics(df, e_orig, e_dest, 'plagiarism')
    print("")
    print(d["plagiarism"])


def test_task():
    model_name = "4-small_br"

    s1, s2 = styles_names(model_name)[0]
    transferred_path = get_transferred_path(s1, s2, model_name)
    metrics_path = get_metrics_dir(transferred_path)
    char_path = get_characteristics_path(model_name)

    calculate_metrics(transferred_path, metrics_path, None, None)