import os.path

import pytest

from dodo import do_evaluation, styles_names, audio_generation
from evaluation.evaluation import *
from evaluation.metrics.intervals import get_interval_distribution_params
from model.colab_tension_vae.params import init
from utils.files_utils import data_tests_path, load_pickle, data_path, get_eval_dir, get_transferred_path, \
    get_metrics_dir, get_characteristics_path, get_audios_path


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
    trans_path = get_transferred_path(s1, s2, model_name)
    metrics = load_pickle(get_metrics_dir(trans_path))

    eval_path = get_eval_dir(transferred_path=trans_path)

    plot_intervals_improvements(orig="Bach", dest="ragtime", interval_distances=metrics['intervals'],
                                plot_path=eval_path)
    plot_intervals_improvements(orig="ragtime", dest="Bach", interval_distances=metrics['intervals'],
                                plot_path=eval_path)


def test_intervals_results():
    d = {'Style': 2 * (5 * ['a'] + 5 * ['b']),
         'target': 2 * (5 * ['b'] + 5 * ['a']),
         'type': 10 * ["log(d(m',s')/d(m,s')) (< 0)\n Got closer to the new style"]
                 + 10 * ["log(d(m',s)/d(m,s)) (> 0)\n Got away from the old style"],
         'value': 5 * [-2] + [4, -1, -1, 4, 4] + [-3, -3, 2, -3, -3] + 5 * [-1]
         }
    d = pd.DataFrame(d)
    eval_path = f"{data_path}/debug_outputs/Evaluation"
    df_results = get_bigrams_results(d, 'a', 'b', eval_path, "test_intervals_results")

    assert list(df_results[df_results["Transference"] == f"a to b"]["% got closer"])[0] == 100
    assert list(df_results[df_results["Transference"] == f"b to a"]["% got closer"])[0] == 40
    assert list(df_results[df_results["Transference"] == f"a to b"]["% got away"])[0] == 20
    assert list(df_results[df_results["Transference"] == f"b to a"]["% got away"])[0] == 0


def test_evaluate_intervals_distribution_small(bmmr_dfs):
    init(4)
    s1, s2, model_name = "Bach", "ragtime", "brmf_4b"
    eval_path = f"{data_path}/debug_outputs/Evaluation"
    metrics = load_pickle(get_metrics_dir(get_transferred_path(s1, s2, model_name)))
    _, table, _ = evaluate_bigrams_distribution(metrics['intervals'], metrics["original_style"],
                                                metrics["target_style"], eval_path, "test_evaluate_intervals_distribution_small")
    print(table)
    table.to_csv(f"{data_path}/debug_outputs/tables/table_intervals-small.csv", index=False)


def test_evaluate_intervals_distribution(all_dfs):
    init(4)
    s1, s2, model_name = "Bach", "ragtime", "brmf_4b"
    eval_path = f"{data_path}/debug_outputs/Evaluation"
    metrics = load_pickle(get_metrics_dir(get_transferred_path(s1, s2, model_name)))
    _, table, _ = evaluate_bigrams_distribution(metrics['intervals'], metrics["original_style"],
                                                metrics["target_style"], eval_path, "test_evaluate_intervals_distribution")
    print(table)
    table.to_csv(f"{data_path}/debug_outputs/tables/table_intervals-all.csv", index=False)


def test_evaluate_all_single_intervals_distribution(all_dfs):
    init(4)
    s1, s2, model_name = "Bach", "ragtime", "brmf_4b"
    eval_path = f"{data_path}/debug_outputs/Evaluation"
    metrics = load_pickle(get_metrics_dir(get_transferred_path(s1, s2, model_name)))
    _, table, _ = evaluate_bigrams_distribution(metrics['intervals'], metrics["original_style"],
                                                metrics["target_style"], eval_path,
                                                "test_evaluate_all_single_intervals_distribution", context='talk')
    print(table)
    table.to_csv(f"{data_path}/debug_outputs/tables/table_intervals-all_single.csv", index=False)


# ----------------------------------------------------- Plagiarism -----------------------------------------------------
def test_calculate_resume_table():
    d = {"Title": ["Cancion 1", "c2", "c3", "c3", "c4", "c5"],
         "Style": ["a", "b", "a", "b", "b", "b"],
         "target": ["b", "a", "b", "a", "a", "a"],
         "value": [1, 2, 4, 1, 4, 5]
         }
    d = pd.DataFrame(d)

    t = calculate_resume_table(d, 1)
    assert list(t["Style"]) == ["a", "b"]
    assert list(t["Target"]) == ["b", "a"]
    assert list(t["Percentage of winners"]) == [0.5, 0.25]

    t = calculate_resume_table(d, 2)
    assert list(t["Percentage of winners"]) == [0.5, 0.5]


def test_evaluate_plagiarism_1():
    init(4)
    model_name = '4-small_br'
    s1, s2 = "Bach", "ragtime"
    metrics = load_pickle(f"{data_path}{model_name}/embeddings/df_transferred_{s1}_{s2}-metrics.pkl")
    eval_path = f"{data_path}/debug_outputs/Evaluation"

    _, table, _ = evaluate_plagiarism(metrics["plagiarism"], None, None, eval_path)
    print(table)


def test_evaluate_plagiarism_separated_2():
    init(4)
    model_name = '4-small_br'
    s1, s2 = "Bach", "ragtime"
    metrics = load_pickle(f"{data_path}{model_name}/embeddings/df_transferred_{s1}_{s2}-metrics.pkl")
    cache_path = f"{data_path}/debug_outputs/tables/table_plagiarism-all_separated-2"
    eval_path = f"{data_path}/debug_outputs/Evaluation"

    _, table, _ = evaluate_plagiarism(metrics["plagiarism"], None, None, eval_path, thold=2)

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
    eval_path = f"{data_path}/debug_outputs/Evaluation"

    _, table, _ = evaluate_plagiarism(metrics["plagiarism"], None, None, eval_path, thold=0.1)

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
    eval_path = f"{data_path}/debug_outputs/Evaluation"

    _, table, _ = evaluate_plagiarism(metrics["plagiarism"], None, None, eval_path, thold=0.25)

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
    eval_path = f"{data_path}/debug_outputs/Evaluation"

    _, table, _ = evaluate_plagiarism(metrics["plagiarism"], None, None, eval_path, thold=0.5)

    for s, t in zip(table["Style"], table["Target"]):
        assert s != t

    table.to_csv(cache_path + '.csv', index=False)
    print(table)


def test_evaluate_model():
    init(4)

    s1, s2, model_name = "small_Bach", "small_ragtime", "4-small_br"
    trans_path = get_transferred_path(s1, s2, model_name)
    df = load_pickle(trans_path)

    metrics_dir = get_metrics_dir(trans_path)
    metrics = load_pickle(f"{metrics_dir}/metrics_{s1}_to_{s2}")

    styles_path = get_characteristics_path(model_name)
    styles = load_pickle(styles_path)

    eval_dir = get_eval_dir(trans_path)
    melodic_musicality_distribution = load_pickle(eval_dir + '/melodic_distribution.pkl')
    rhythmic_musicality_distribution = load_pickle(eval_dir + '/rhythmic_distribution.pkl')

    evaluate_model(df, metrics, styles, melodic_musicality_distribution, rhythmic_musicality_distribution,
                   f"{data_path}/debug_outputs/", thold=2)


def test_evaluation_task():
    init(4)
    # model_name = "4-small_br"
    model_name = "brmf_4b"

    styles_path = get_characteristics_path(model_name)

    for style1, style2 in styles_names(model_name):
        transferred_path = get_transferred_path(style1, style2, model_name)
        eval_path = get_eval_dir(transferred_path)

        do_evaluation(transferred_path, styles_path, eval_path, style1, style2)


def test_audio_generation():
    model_name = "brmf_4b"
    s1 = "Mozart"
    s2 = "ragtime"
    suffix = f'{s1}_to_{s2}'

    audios_path = get_audios_path(model_name)
    transferred_path = get_transferred_path(s1, s2, model_name)

    eval_dir = get_eval_dir(transferred_path)
    successful_rolls_prefix = f"{eval_dir}/successful_rolls-"

    audio_generation(transferred_path, audios_path, successful_rolls_prefix, suffix, s1, s2)
