import os.path

import numpy as np
import pytest

from dodo import do_evaluation, styles_names
from evaluation.evaluation import *
from evaluation.metrics.intervals import get_interval_distribution_params
from model.colab_tension_vae.params import init
from utils.files_utils import data_tests_path, load_pickle, data_path, get_eval_dir, get_transferred_path, \
    get_metrics_dir, get_characteristics_path


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
    df1 = load_pickle(os.path.join(data_path, "brmf_4b/embeddings/df_transferred_Bach_ragtime.pkl"))
    df2 = load_pickle(os.path.join(data_path, "brmf_4b/embeddings/df_transferred_ragtime_Bach.pkl"))
    return pd.concat([df1, df2], axis=0)


@pytest.fixture
def all_dfs():
    df1 = load_pickle(os.path.join(data_path, "brmf_4b/embeddings/df_transferred_Bach_ragtime.pkl"))
    df2 = load_pickle(os.path.join(data_path, "brmf_4b/embeddings/df_transferred_ragtime_Bach.pkl"))
    df_br = pd.concat([df1, df2], axis=0)

    df1 = load_pickle(os.path.join(data_path, "brmf_4b/embeddings/df_transferred_Bach_Frescobaldi.pkl"))
    df2 = load_pickle(os.path.join(data_path, "brmf_4b/embeddings/df_transferred_Frescobaldi_Bach.pkl"))
    df_bf = pd.concat([df1, df2], axis=0)

    df1 = load_pickle(os.path.join(data_path, "brmf_4b/embeddings/df_transferred_Bach_Mozart.pkl"))
    df2 = load_pickle(os.path.join(data_path, "brmf_4b/embeddings/df_transferred_Mozart_Bach.pkl"))
    df_bm = pd.concat([df1, df2], axis=0)

    df1 = load_pickle(os.path.join(data_path, "brmf_4b/embeddings/df_transferred_Frescobaldi_ragtime.pkl"))
    df2 = load_pickle(os.path.join(data_path, "brmf_4b/embeddings/df_transferred_ragtime_Frescobaldi.pkl"))
    df_fr = pd.concat([df1, df2], axis=0)

    df1 = load_pickle(os.path.join(data_path, "brmf_4b/embeddings/df_transferred_Frescobaldi_Mozart.pkl"))
    df2 = load_pickle(os.path.join(data_path, "brmf_4b/embeddings/df_transferred_Mozart_Frescobaldi.pkl"))
    df_fm = pd.concat([df1, df2], axis=0)

    df1 = load_pickle(os.path.join(data_path, "brmf_4b/embeddings/df_transferred_Mozart_ragtime.pkl"))
    df2 = load_pickle(os.path.join(data_path, "brmf_4b/embeddings/df_transferred_ragtime_Mozart.pkl"))
    df_mr = pd.concat([df1, df2], axis=0)

    return [df_br, df_bf, df_bm, df_fr, df_fm, df_mr]


@pytest.fixture
def bmmr_dfs():
    df1 = load_pickle(os.path.join(data_path, "brmf_4b/embeddings/df_transferred_Mozart_ragtime.pkl"))
    df2 = load_pickle(os.path.join(data_path, "brmf_4b/embeddings/df_transferred_ragtime_Mozart.pkl"))
    df_mr = pd.concat([df1, df2], axis=0)

    df1 = load_pickle(os.path.join(data_path, "brmf_4b/embeddings/df_transferred_Bach_Mozart.pkl"))
    df2 = load_pickle(os.path.join(data_path, "brmf_4b/embeddings/df_transferred_Mozart_Bach.pkl"))
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
    s1, s2, model_name = "Bach", "ragtime", "brmf_4b"
    metrics = load_pickle(get_metrics_dir(get_transferred_path(s1, s2, model_name)))
    plot_intervals_distribution(orig="Bach", dest="ragtime", interval_distances=metrics['intervals'])
    plot_intervals_distribution(orig="ragtime", dest="Bach", interval_distances=metrics['intervals'])


def test_intervals_results():
    d = {'Style': 2 * (5 * ['a'] + 5 * ['b']),
         'target': 2 * (5 * ['b'] + 5 * ['a']),
         'type': 10 * ["log(d(m',s')/d(m,s')) (< 0)\n Got closer to the new style"]
                 + 10 * ["log(d(m',s)/d(m,s)) (> 0)\n Got away from the old style"],
         'value': 5 * [-2] + [4, -1, -1, 4, 4] + [-3, -3, 2, -3, -3] + 5 * [-1]
         }
    df = pd.DataFrame(d)
    df_results = get_intervals_results(df, 'a', 'b')

    assert list(df_results[df_results["Transference"] == f"a to b"]["% got closer"])[0] == 100
    assert list(df_results[df_results["Transference"] == f"b to a"]["% got closer"])[0] == 40
    assert list(df_results[df_results["Transference"] == f"a to b"]["% got away"])[0] == 20
    assert list(df_results[df_results["Transference"] == f"b to a"]["% got away"])[0] == 0


def test_evaluate_intervals_distribution_small(bmmr_dfs):
    init(4)
    s1, s2, model_name = "Bach", "ragtime", "brmf_4b"
    metrics = load_pickle(get_metrics_dir(get_transferred_path(s1, s2, model_name)))
    _, table, _ = evaluate_bigrams_distribution(metrics['intervals'], metrics["original_style"], metrics["target_style"])
    print(table)
    table.to_csv(f"{data_path}/debug_outputs/tables/table_intervals-small.csv", index=False)


def test_evaluate_intervals_distribution(all_dfs):
    init(4)
    s1, s2, model_name = "Bach", "ragtime", "brmf_4b"
    metrics = load_pickle(get_metrics_dir(get_transferred_path(s1, s2, model_name)))
    _, table, _ = evaluate_bigrams_distribution(metrics['intervals'], metrics["original_style"], metrics["target_style"])
    print(table)
    table.to_csv(f"{data_path}/debug_outputs/tables/table_intervals-all.csv", index=False)


def test_evaluate_all_single_intervals_distribution(all_dfs):
    init(4)
    s1, s2, model_name = "Bach", "ragtime", "brmf_4b"
    metrics = load_pickle(get_metrics_dir(get_transferred_path(s1, s2, model_name)))
    _, table, _ = evaluate_bigrams_distribution(metrics['intervals'], metrics["original_style"], metrics["target_style"], context='talk')
    print(table)
    table.to_csv(f"{data_path}/debug_outputs/tables/table_intervals-all_single.csv", index=False)


# ----------------------------------------------------- Plagiarism -----------------------------------------------------
def test_calculate_resume_table():
    d = {"Title": ["Cancion 1", "c2", "c3", "c3", "c4", "c5"],
         "Style": ["a", "b", "a", "b", "b", "b"],
         "target": ["b", "a", "b", "a", "a", "a"],
         "value": [1, 2, 4, 1, 4, 5]
         }
    df = pd.DataFrame(d)

    t = calculate_resume_table(df, 1)
    assert list(t["Style"]) == ["a", "b"]
    assert list(t["Target"]) == ["b", "a"]
    assert list(t["Percentage of winners"]) == [0.5, 0.25]

    t = calculate_resume_table(df, 2)
    assert list(t["Percentage of winners"]) == [0.5, 0.5]


def test_evaluate_plagiarism_1():
    init(4)
    model_name = '4-small_br'
    s1, s2 = "Bach", "ragtime"
    metrics = load_pickle(f"{data_path}{model_name}/embeddings/df_transferred_{s1}_{s2}-metrics.pkl")
    cache_path = f"{data_path}/debug_outputs/tables/table_plagiarism-small"

    _, table, _ = evaluate_plagiarism(metrics["plagiarism"], None, None)
    print(table)


def test_evaluate_plagiarism_separated_2():
    init(4)
    model_name = '4-small_br'
    s1, s2 = "Bach", "ragtime"
    metrics = load_pickle(f"{data_path}{model_name}/embeddings/df_transferred_{s1}_{s2}-metrics.pkl")
    cache_path = f"{data_path}/debug_outputs/tables/table_plagiarism-all_separated-2"

    _, table, _ = evaluate_plagiarism(metrics["plagiarism"], None, None, thold=2)

    for s, t in zip(table["Style"], table["Target"]):
        assert s != t

    table.to_csv(cache_path + '.csv', index=False)
    print(table)


def test_evaluate_plagiarism_separated_proportional_10():
    init(4)
    model_name = '4-small_br'
    s1, s2 = "Bach", "ragtime"
    metrics = load_pickle(f"{data_path}{model_name}/embeddings/df_transferred_{s1}_{s2}-metrics.pkl")
    cache_path = f"{data_path}/debug_outputs/tables/table_plagiarism-all_separated-proportional_10"

    _, table, _ = evaluate_plagiarism(metrics["plagiarism"], None, None, thold=0.1)

    for s, t in zip(table["Style"], table["Target"]):
        assert s != t

    table.to_csv(cache_path + '.csv', index=False)
    print(table)


def test_evaluate_plagiarism_separated_proportional_25():
    init(4)
    model_name = '4-small_br'
    s1, s2 = "Bach", "ragtime"
    metrics = load_pickle(f"{data_path}{model_name}/embeddings/df_transferred_{s1}_{s2}-metrics.pkl")
    cache_path = f"{data_path}/debug_outputs/tables/table_plagiarism-all_separated-proportional_25"

    _, table, _ = evaluate_plagiarism(metrics["plagiarism"], None, None, thold=0.25)

    for s, t in zip(table["Style"], table["Target"]):
        assert s != t

    table.to_csv(cache_path + '.csv', index=False)
    print(table)


def test_evaluate_plagiarism_separated_proportional_50():
    init(4)
    model_name = '4-small_br'
    s1, s2 = "Bach", "ragtime"
    metrics = load_pickle(f"{data_path}{model_name}/embeddings/df_transferred_{s1}_{s2}-metrics.pkl")
    cache_path = f"{data_path}/debug_outputs/tables/table_plagiarism-all_separated-proportional_5"

    _, table, _ = evaluate_plagiarism(metrics["plagiarism"], None, None, thold=0.5)

    for s, t in zip(table["Style"], table["Target"]):
        assert s != t

    table.to_csv(cache_path + '.csv', index=False)
    print(table)


def test_display_best_audios():
    init(4)

    s1, s2, model_name = "Bach", "ragtime", "brmf_4b"
    metrics = load_pickle(get_metrics_dir(get_transferred_path(s1, s2, model_name)))

    evaluate_model(metrics, {}, f"{data_path}/debug_outputs/audios/successful",
                   cache_path=f"{data_path}/debug_outputs/tables/table_plagiarism-all_separated-2", merge=False,
                   by_distance=True, thold=2)


def test_evaluation_task():
    init(4)
    model_name = "4-small_br"

    styles_path = get_characteristics_path(model_name)

    for style1, style2 in styles_names(model_name):
        transferred_path = get_transferred_path(style1, style2, model_name)
        eval_path = get_eval_dir(transferred_path)

        do_evaluation(transferred_path, styles_path, eval_path, style1, style2)