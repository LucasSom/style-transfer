import os.path

import numpy as np
import pandas as pd
import pytest

from evaluation.evaluation import evaluate_single_intervals_distribution, evaluate_multiple_intervals_distribution, \
    evaluate_single_plagiarism, evaluate_multiple_plagiarism, get_intervals_results
from evaluation.metrics.intervals import get_interval_distribution_params
from model.colab_tension_vae.params import init
from utils.files_utils import data_tests_path, load_pickle, data_path, save_pickle


@pytest.fixture
def confusion_matrices():
    return [np.loadtxt(data_tests_path + "interval_confusion_matrices/1.csv", encoding='utf-16', delimiter=",",
                       dtype=int),
            np.loadtxt(data_tests_path + "interval_confusion_matrices/2.csv", encoding='utf-16', delimiter=",",
                       dtype=int),
            np.loadtxt(data_tests_path + "interval_confusion_matrices/3.csv", encoding='utf-16', delimiter=",",
                       dtype=int)
            ]


@pytest.fixture
def df_transferred():
    df1 = load_pickle(os.path.join(data_path, "embeddings/brmf_4b/df_transferred_Bach_ragtime.pkl"))
    df2 = load_pickle(os.path.join(data_path, "embeddings/brmf_4b/df_transferred_ragtime_Bach.pkl"))
    return pd.concat([df1, df2], axis=0)


@pytest.fixture
def all_dfs():
    df1 = load_pickle(os.path.join(data_path, "embeddings/brmf_4b/df_transferred_Bach_ragtime.pkl"))
    df2 = load_pickle(os.path.join(data_path, "embeddings/brmf_4b/df_transferred_ragtime_Bach.pkl"))
    df_br = pd.concat([df1, df2], axis=0)

    df1 = load_pickle(os.path.join(data_path, "embeddings/brmf_4b/df_transferred_Bach_Frescobaldi.pkl"))
    df2 = load_pickle(os.path.join(data_path, "embeddings/brmf_4b/df_transferred_Frescobaldi_Bach.pkl"))
    df_bf = pd.concat([df1, df2], axis=0)

    df1 = load_pickle(os.path.join(data_path, "embeddings/brmf_4b/df_transferred_Bach_Mozart.pkl"))
    df2 = load_pickle(os.path.join(data_path, "embeddings/brmf_4b/df_transferred_Mozart_Bach.pkl"))
    df_bm = pd.concat([df1, df2], axis=0)

    df1 = load_pickle(os.path.join(data_path, "embeddings/brmf_4b/df_transferred_Frescobaldi_ragtime.pkl"))
    df2 = load_pickle(os.path.join(data_path, "embeddings/brmf_4b/df_transferred_ragtime_Frescobaldi.pkl"))
    df_fr = pd.concat([df1, df2], axis=0)

    df1 = load_pickle(os.path.join(data_path, "embeddings/brmf_4b/df_transferred_Frescobaldi_Mozart.pkl"))
    df2 = load_pickle(os.path.join(data_path, "embeddings/brmf_4b/df_transferred_Mozart_Frescobaldi.pkl"))
    df_fm = pd.concat([df1, df2], axis=0)

    df1 = load_pickle(os.path.join(data_path, "embeddings/brmf_4b/df_transferred_Mozart_ragtime.pkl"))
    df2 = load_pickle(os.path.join(data_path, "embeddings/brmf_4b/df_transferred_ragtime_Mozart.pkl"))
    df_mr = pd.concat([df1, df2], axis=0)

    return [df_br, df_bf, df_bm, df_fr, df_fm, df_mr]


@pytest.fixture
def bmmr_dfs():
    df1 = load_pickle(os.path.join(data_path, "embeddings/brmf_4b/df_transferred_Mozart_ragtime.pkl"))
    df2 = load_pickle(os.path.join(data_path, "embeddings/brmf_4b/df_transferred_ragtime_Mozart.pkl"))
    df_mr = pd.concat([df1, df2], axis=0)

    df1 = load_pickle(os.path.join(data_path, "embeddings/brmf_4b/df_transferred_Bach_Mozart.pkl"))
    df2 = load_pickle(os.path.join(data_path, "embeddings/brmf_4b/df_transferred_Mozart_Bach.pkl"))
    df_bm = pd.concat([df1, df2], axis=0)

    return [df_bm, df_mr]


def test_intervals_distributions():
    # intervals = [1, 0, 2, 1, 0, -3, -1]
    intervals = [1, -1, 1, -1]
    m, s = get_interval_distribution_params(intervals)

    assert m == 0
    assert s == 1


def test_intervals_characteristic_confusion_matrix(confusion_matrices):
    avg = np.average(confusion_matrices, axis=0)
    avg_test = np.loadtxt(data_tests_path + "interval_confusion_matrices/avg.csv", encoding='utf-16', delimiter=",",
                          dtype=int)

    for i, j in zip(range(avg.shape[0]), range(avg.shape[1])):
        assert avg[i, j] == avg_test[i, j]


# --------------------------------------------------- Intervals plots --------------------------------------------------
def test_evaluate_single_intervals_distribution(df_transferred):
    init(4)
    evaluate_single_intervals_distribution(df_transferred, orig="Bach", dest="ragtime")
    evaluate_single_intervals_distribution(df_transferred, orig="ragtime", dest="Bach")


def test_intervals_results():
    d = {'orig': 2 * (5 * ['a'] + 5 * ['b']),
         'target': 2 * (5 * ['b'] + 5 * ['a']),
         'type': 10 * ["log(d(ms')/d(ms)) (> 0)\n Got closer to the new style"]
                + 10 * ["log(d(m's')/d(ms')) (< 0)\n Got away from the old style"],
         'value': 5 * [1] + [1, -1, -1, 1, 1] + [-1, -1, 1, -1, -1] + 5 * [1]
         }
    df = pd.DataFrame(d)
    results = {}
    get_intervals_results(df, results, 'a', 'b')

    assert results[f"a to b got closer"] == 1
    assert results[f"b to a got closer"] == 3 / 5
    assert results[f"a to b got away"] == 4 / 5
    assert results[f"b to a got away"] == 0


def test_evaluate_intervals_distribution_small(bmmr_dfs):
    init(4)
    _, _, table = evaluate_multiple_intervals_distribution(bmmr_dfs, True)
    print(table)
    table.to_csv(f"{data_path}/debug_outputs/table_intervals-small.csv")


def test_evaluate_intervals_distribution(all_dfs):
    init(4)
    _, _, table = evaluate_multiple_intervals_distribution(all_dfs, True)
    print(table)
    table.to_csv(f"{data_path}/debug_outputs/table_intervals-all.csv")


def test_evaluate_all_single_intervals_distribution(all_dfs):
    init(4)
    _, _, table = evaluate_multiple_intervals_distribution(all_dfs, False, context='talk')
    print(table)
    table.to_csv(f"{data_path}/debug_outputs/table_intervals-all_single.csv")


# ----------------------------------------------------- Plagiarism -----------------------------------------------------
def test_evaluate_single_plagiarism(df_transferred):
    init(4)
    df1 = evaluate_single_plagiarism(df_transferred, orig="Bach", dest="ragtime")
    df2 = evaluate_single_plagiarism(df_transferred, orig="ragtime", dest="Bach")
    df1.to_csv(f"{data_path}/debug_outputs/plagiarism_ranking_table1.csv")
    df2.to_csv(f"{data_path}/debug_outputs/plagiarism_ranking_table2.csv")


def test_evaluate_plagiarism_small(bmmr_dfs):
    init(4)
    evaluate_multiple_plagiarism(bmmr_dfs, True)


def test_evaluate_plagiarism_all(all_dfs):
    init(4)
    evaluate_multiple_plagiarism(all_dfs, True)


def test_evaluate_plagiarism_separated(all_dfs):
    init(4)
    evaluate_multiple_plagiarism(all_dfs, False)
