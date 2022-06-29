import numpy as np
import pytest

from evaluation.metrics.intervals import get_interval_distribution
from utils.files_utils import data_tests_path


@pytest.fixture
def confusion_matrices():
    return [np.loadtxt(data_tests_path + "interval_confusion_matrices/1.csv", delimiter=",", dtype=int),
            np.loadtxt(data_tests_path + "interval_confusion_matrices/2.csv", delimiter=",", dtype=int),
            np.loadtxt(data_tests_path + "interval_confusion_matrices/3.csv", delimiter=",", dtype=int)
            ]


def test_intervals_distributions():
    # intervals = [1, 0, 2, 1, 0, -3, -1]
    intervals = [1, -1, 1, -1]
    m, s = get_interval_distribution(intervals)

    assert m == 0
    assert s == 1


def test_intervals_characteristic_confusion_matrix(confusion_matrices):
    avg = np.average(confusion_matrices, axis=1)
    avg_test = np.loadtxt(data_tests_path + "interval_confusion_matrices/avg.csv", delimiter=",", dtype=int)

    assert avg == avg_test

