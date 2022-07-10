import numpy as np
import pytest

from evaluation.metrics.intervals import get_interval_distribution
from utils.files_utils import data_tests_path


@pytest.fixture
def confusion_matrices():
    return [np.loadtxt(data_tests_path + "interval_confusion_matrices/1.csv", encoding='utf-16', delimiter=",",
                       dtype=int),
            np.loadtxt(data_tests_path + "interval_confusion_matrices/2.csv", encoding='utf-16', delimiter=",",
                       dtype=int),
            np.loadtxt(data_tests_path + "interval_confusion_matrices/3.csv", encoding='utf-16', delimiter=",",
                       dtype=int)
            ]


def test_intervals_distributions():
    # intervals = [1, 0, 2, 1, 0, -3, -1]
    intervals = [1, -1, 1, -1]
    m, s = get_interval_distribution(intervals)

    assert m == 0
    assert s == 1


def my_avg(ms):
    r, c = ms[0].shape
    avg_matrix = np.zeros((r, c))
    for i in range(r):
        for j in range(c):
            avg = 0
            for m in ms:
                avg += m[i, j]
            avg_matrix[i, j] = avg / len(ms)
    return avg_matrix


def test_intervals_characteristic_confusion_matrix(confusion_matrices):
    # avg = np.average(confusion_matrices, axis=1)
    avg = my_avg(confusion_matrices)
    avg_test = np.loadtxt(data_tests_path + "interval_confusion_matrices/avg.csv",
                          encoding='utf-16', delimiter=",", dtype=int)

    for i, j in zip(range(avg.shape[0]), range(avg.shape[1])):
        assert avg[i, j] == avg_test[i, j]

