import os.path

import pytest

from dodo import do_evaluation, transference_names, audio_generation, do_overall_single_evaluation, models, \
    mixture_models, \
    sheets_generation, create_html, create_examples, style_names, do_family_evaluation, model_families, mutations
from evaluation.evaluation import *
from evaluation.metrics.intervals import get_interval_distribution_params
from model.colab_tension_vae.params import init
from utils.files_utils import data_tests_path, load_pickle, data_path, get_eval_dir, get_transferred_path, \
    get_metrics_dir, get_characteristics_path, get_audios_path, save_pickle, get_sheets_path, get_examples_path
from evaluation.overall_evaluation import get_packed_metrics
from utils.plots_utils import plot_intervals_improvements


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
    df1 = load_pickle(os.path.join(data_path, "models/brmf_4b/embeddings/df_transferred_Bach_ragtime.pkl"))
    df2 = load_pickle(os.path.join(data_path, "models/brmf_4b/embeddings/df_transferred_ragtime_Bach.pkl"))
    return pd.concat([df1, df2], axis=0)


@pytest.fixture
def all_dfs():
    df1 = load_pickle(os.path.join(data_path, "models/brmf_4b/embeddings/df_transferred_Bach_ragtime.pkl"))
    df2 = load_pickle(os.path.join(data_path, "models/brmf_4b/embeddings/df_transferred_ragtime_Bach.pkl"))
    df_br = pd.concat([df1, df2], axis=0)

    df1 = load_pickle(os.path.join(data_path, "models/brmf_4b/embeddings/df_transferred_Bach_Frescobaldi.pkl"))
    df2 = load_pickle(os.path.join(data_path, "models/brmf_4b/embeddings/df_transferred_Frescobaldi_Bach.pkl"))
    df_bf = pd.concat([df1, df2], axis=0)

    df1 = load_pickle(os.path.join(data_path, "models/brmf_4b/embeddings/df_transferred_Bach_Mozart.pkl"))
    df2 = load_pickle(os.path.join(data_path, "models/brmf_4b/embeddings/df_transferred_Mozart_Bach.pkl"))
    df_bm = pd.concat([df1, df2], axis=0)

    df1 = load_pickle(os.path.join(data_path, "models/brmf_4b/embeddings/df_transferred_Frescobaldi_ragtime.pkl"))
    df2 = load_pickle(os.path.join(data_path, "models/brmf_4b/embeddings/df_transferred_ragtime_Frescobaldi.pkl"))
    df_fr = pd.concat([df1, df2], axis=0)

    df1 = load_pickle(os.path.join(data_path, "models/brmf_4b/embeddings/df_transferred_Frescobaldi_Mozart.pkl"))
    df2 = load_pickle(os.path.join(data_path, "models/brmf_4b/embeddings/df_transferred_Mozart_Frescobaldi.pkl"))
    df_fm = pd.concat([df1, df2], axis=0)

    df1 = load_pickle(os.path.join(data_path, "models/brmf_4b/embeddings/df_transferred_Mozart_ragtime.pkl"))
    df2 = load_pickle(os.path.join(data_path, "models/brmf_4b/embeddings/df_transferred_ragtime_Mozart.pkl"))
    df_mr = pd.concat([df1, df2], axis=0)

    return [df_br, df_bf, df_bm, df_fr, df_fm, df_mr]


@pytest.fixture
def bmmr_dfs():
    df1 = load_pickle(os.path.join(data_path, "models/brmf_4b/embeddings/df_transferred_Mozart_ragtime.pkl"))
    df2 = load_pickle(os.path.join(data_path, "models/brmf_4b/embeddings/df_transferred_ragtime_Mozart.pkl"))
    df_mr = pd.concat([df1, df2], axis=0)

    df1 = load_pickle(os.path.join(data_path, "models/brmf_4b/embeddings/df_transferred_Bach_Mozart.pkl"))
    df2 = load_pickle(os.path.join(data_path, "models/brmf_4b/embeddings/df_transferred_Mozart_Bach.pkl"))
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
    metrics_dir = get_metrics_dir(model_name)
    metrics = load_pickle(metrics_dir + '/metrics_Bach_to_ragtime')

    eval_path = get_eval_dir(model_name)

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
    # assert list(df_results[df_results["Transference"] == f"b to a"]["% got closer"])[0] == 40
    assert list(df_results[df_results["Transference"] == f"a to b"]["% got away"])[0] == 20
    # assert list(df_results[df_results["Transference"] == f"b to a"]["% got away"])[0] == 0


def test_evaluate_intervals_distribution_small(bmmr_dfs):
    init(4)
    s1, s2, model_name = "Bach", "ragtime", "brmf_4b"
    eval_path = f"{data_path}/debug_outputs/Evaluation"
    metrics_dir = get_metrics_dir(model_name)
    metrics = load_pickle(metrics_dir + '/metrics_Bach_to_ragtime')
    table, _ = evaluate_bigrams_distribution(metrics['intervals'], metrics["original_style"],
                                             metrics["target_style"], eval_path,
                                             "test_evaluate_intervals_distribution_small")
    print(table)
    table.to_csv(f"{data_path}/debug_outputs/tables/table_intervals-small.csv", index=False)


def test_evaluate_intervals_distribution(all_dfs):
    init(4)
    s1, s2, model_name = "Bach", "ragtime", "brmf_4b"
    eval_path = f"{data_path}/debug_outputs/Evaluation"
    metrics_dir = get_metrics_dir(model_name)
    metrics = load_pickle(metrics_dir + '/metrics_Bach_to_ragtime')
    table, _ = evaluate_bigrams_distribution(metrics['intervals'], metrics["original_style"],
                                             metrics["target_style"], eval_path,
                                             "test_evaluate_intervals_distribution")
    print(table)
    table.to_csv(f"{data_path}/debug_outputs/tables/table_intervals-all.csv", index=False)


def test_evaluate_all_single_intervals_distribution(all_dfs):
    init(4)
    s1, s2, model_name = "Bach", "ragtime", "brmf_4b"
    eval_path = f"{data_path}/debug_outputs/Evaluation"
    metrics_dir = get_metrics_dir(model_name)
    metrics = load_pickle(metrics_dir + '/metrics_Bach_to_ragtime')
    table, _ = evaluate_bigrams_distribution(metrics['intervals'], metrics["original_style"],
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
    assert list(t["Percentage of winners"]) == [50, 25]

    t = calculate_resume_table(d, 2)
    assert list(t["Percentage of winners"]) == [50, 50]


def test_evaluate_plagiarism_1():
    init(4)
    model_name = '4-small_br'
    mutation = "Mutation_add_1"
    s1, s2 = "Bach", "ragtime"
    metrics = load_pickle(f"{data_path}models/{model_name}/metrics/metrics_small_{s1}_to_small_{s2}.pkl")
    eval_path = f"{data_path}/debug_outputs/Evaluation"

    table, _, _ = evaluate_plagiarism(metrics["plagiarism-dist"], None, None, mutation, eval_path)
    print(table)

    table, _, _ = evaluate_plagiarism(metrics["plagiarism-diff"], None, None, mutation, eval_path)
    print(table)


def test_evaluate_plagiarism_separated_2():
    init(4)
    model_name = '4-small_br'
    mutation = "Mutation_add_1"
    s1, s2 = "Bach", "ragtime"
    metrics = load_pickle(f"{data_path}models/{model_name}/metrics/metrics_small_{s1}_to_small_{s2}.pkl")
    cache_path = f"{data_path}/debug_outputs/tables/table_plagiarism-all_separated-2"
    eval_path = f"{data_path}/debug_outputs/Evaluation"

    table, _, _ = evaluate_plagiarism(metrics["plagiarism-dist"], None, None, mutation, eval_path, thold=2)

    for s, t in zip(table["Style"], table["Target"]):
        assert s != t

    table.to_csv(cache_path + '-dist.csv', index=False)
    print(table)

    table, _, _ = evaluate_plagiarism(metrics["plagiarism-diff"], None, None, mutation, eval_path, thold=2)

    for s, t in zip(table["Style"], table["Target"]):
        assert s != t

    table.to_csv(cache_path + '-diff.csv', index=False)
    print(table)


def test_evaluate_plagiarism_separated_proportional_10():
    init(4)
    model_name = '4-small_br'
    mutation = "Mutation_add_1"
    s1, s2 = "Bach", "ragtime"
    metrics = load_pickle(f"{data_path}models/{model_name}/metrics/metrics_small_{s1}_to_small_{s2}.pkl")
    cache_path = f"{data_path}/debug_outputs/tables/table_plagiarism-all_separated-proportional_10"
    eval_path = f"{data_path}/debug_outputs/Evaluation"

    table, _, _ = evaluate_plagiarism(metrics["plagiarism-dist"], None, None, mutation, eval_path, thold=0.1)

    for s, t in zip(table["Style"], table["Target"]):
        assert s != t

    table.to_csv(cache_path + '.csv', index=False)
    print(table)


def test_evaluate_plagiarism_separated_proportional_25():
    init(4)
    model_name = '4-small_br'
    mutation = "Mutation_add_1"
    s1, s2 = "Bach", "ragtime"
    metrics = load_pickle(f"{data_path}models/{model_name}/metrics/metrics_small_{s1}_to_small_{s2}.pkl")
    cache_path = f"{data_path}/debug_outputs/tables/table_plagiarism-all_separated-proportional_25"
    eval_path = f"{data_path}/debug_outputs/Evaluation"

    table, _, _ = evaluate_plagiarism(metrics["plagiarism-diff"], None, None, mutation, eval_path, thold=0.25)

    for s, t in zip(table["Style"], table["Target"]):
        assert s != t

    table.to_csv(cache_path + '.csv', index=False)
    print(table)


def test_evaluate_plagiarism_separated_proportional_50():
    init(4)
    model_name = '4-small_br'
    mutation = "Mutation_add_1"
    s1, s2 = "Bach", "ragtime"
    metrics = load_pickle(f"{data_path}models/{model_name}/metrics/metrics_small_{s1}_to_small_{s2}.pkl")
    cache_path = f"{data_path}/debug_outputs/tables/table_plagiarism-all_separated-proportional_5"
    eval_path = f"{data_path}/debug_outputs/Evaluation"

    table, _, _ = evaluate_plagiarism(metrics["plagiarism-dist"], None, None, mutation, eval_path, thold=0.5)

    for s, t in zip(table["Style"], table["Target"]):
        assert s != t

    table.to_csv(cache_path + '.csv', index=False)
    print(table)


def test_evaluate_model():
    init(4)

    s1, s2, model_name = "small_Bach", "small_ragtime", "4-small_br"
    mutation = "Mutation_add_sub_1"
    trans_path = get_transferred_path(s1, s2, model_name)
    df = load_pickle(trans_path)

    metrics_dir = get_metrics_dir(model_name)
    metrics = load_pickle(f"{metrics_dir}/metrics_{mutation}_{s1}_to_{s2}")

    styles_path = get_characteristics_path(model_name)
    styles = load_pickle(styles_path)

    melodic_musicality_distribution = load_pickle(data_path + 'data_analysis/melodic_distribution.pkl')
    rhythmic_musicality_distribution = load_pickle(data_path + 'data_analysis/rhythmic_distribution.pkl')

    evaluate_model(df, metrics, styles, melodic_musicality_distribution, rhythmic_musicality_distribution, mutation,
                   f"{data_path}/debug_outputs/", thold=2)


def test_evaluation_task_4br():
    init(4)
    # model_name = "4-small_br"
    model_name = "4-br-96"
    mutation = "Mutation_add_sub_1"

    styles_path = get_characteristics_path(model_name)
    metrics_dir = get_metrics_dir(model_name)
    eval_path = get_eval_dir(model_name)

    for style1, style2 in transference_names(model_name):
        transferred_path = get_transferred_path(style1, style2, model_name)

        do_evaluation(transferred_path, styles_path, metrics_dir, eval_path, style1, style2, mutation)


def test_evaluation_task():
    init(4)
    # model_name = "4-small_br"
    model_name = "brmf_4b_beta-96"
    mutation = "Mutation_add_sub_1"

    styles_path = get_characteristics_path(model_name)
    metrics_dir = get_metrics_dir(model_name)
    eval_path = get_eval_dir(model_name)

    for style1, style2 in transference_names(model_name):
        transferred_path = get_transferred_path(style1, style2, model_name)

        do_evaluation(transferred_path, styles_path, metrics_dir, eval_path, style1, style2, mutation)


def test_evaluation_mixture_model():
    init(4)
    model_name = "4-Lakh_Kern-96"
    mutation = "Mutation_add_1"

    styles_path = get_characteristics_path(model_name)
    metrics_dir = get_metrics_dir(model_name)
    eval_path = get_eval_dir(model_name)

    style1, style2 = "Bach", "Mozart"
    transferred_path = get_transferred_path(style1, style2, model_name)

    do_evaluation(transferred_path, styles_path, metrics_dir, eval_path, style1, style2, mutation)


def test_audio_generation_mixture_model():
    model_name = "4-Lakh_Kern-96"
    s1 = "Bach"
    s2 = "Mozart"
    mutation = "Mutation_add_1"

    audios_path = get_audios_path(model_name)

    transferred_path = get_transferred_path(s1, s2, model_name)
    eval_dir = get_eval_dir(model_name)
    transformation = f'{s1}_to_{s2}'
    successful_rolls_prefix = f"{eval_dir}/successful_rolls-{mutation}"

    audio_generation(mutation, eval_dir, transferred_path, audios_path, successful_rolls_prefix, transformation)


def test_packed_metrics():
    d01 = {"Plagiarism-dist": 1, "Plagiarism-diff": 1, "Musicality": 1, "orig": 's0', "target": 's1',
           "Style": {'s1': 1, 's2': 1, 's3': 1}}
    d02 = {"Plagiarism-dist": 2, "Plagiarism-diff": 2, "Musicality": 2, "orig": 's0', "target": 's2',
           "Style": {'s1': 2, 's2': 2, 's3': 2}}
    d03 = {"Plagiarism-dist": 3, "Plagiarism-diff": 3, "Musicality": 3, "orig": 's0', "target": 's3',
           "Style": {'s1': 3, 's2': 3, 's3': 3}}
    d10 = {"Plagiarism-dist": 10, "Plagiarism-diff": 10, "Musicality": 10, "orig": 's1', "target": 's0',
           "Style": {'s1': 0, 's2': 0, 's3': 0}}
    d12 = {"Plagiarism-dist": 12, "Plagiarism-diff": 12, "Musicality": 12, "orig": 's1', "target": 's2',
           "Style": {'s1': 2, 's2': 2, 's3': 2}}
    d13 = {"Plagiarism-dist": 13, "Plagiarism-diff": 13, "Musicality": 13, "orig": 's1', "target": 's3',
           "Style": {'s1': 3, 's2': 3, 's3': 3}}
    d20 = {"Plagiarism-dist": 20, "Plagiarism-diff": 20, "Musicality": 20, "orig": 's2', "target": 's0',
           "Style": {'s1': 0, 's2': 0, 's3': 0}}
    d21 = {"Plagiarism-dist": 21, "Plagiarism-diff": 21, "Musicality": 21, "orig": 's2', "target": 's1',
           "Style": {'s1': 1, 's2': 1, 's3': 1}}
    d23 = {"Plagiarism-dist": 23, "Plagiarism-diff": 23, "Musicality": 23, "orig": 's2', "target": 's3',
           "Style": {'s1': 3, 's2': 3, 's3': 3}}
    d30 = {"Plagiarism-dist": 30, "Plagiarism-diff": 30, "Musicality": 30, "orig": 's3', "target": 's0',
           "Style": {'s1': 0, 's2': 0, 's3': 0}}
    d31 = {"Plagiarism-dist": 31, "Plagiarism-diff": 31, "Musicality": 31, "orig": 's3', "target": 's1',
           "Style": {'s1': 1, 's2': 1, 's3': 1}}
    d32 = {"Plagiarism-dist": 32, "Plagiarism-diff": 32, "Musicality": 32, "orig": 's3', "target": 's2',
           "Style": {'s1': 2, 's2': 2, 's3': 2}}

    p01 = f'{data_path}tests/overall_metrics_dict-01'
    p02 = f'{data_path}tests/overall_metrics_dict-02'
    p03 = f'{data_path}tests/overall_metrics_dict-03'
    p10 = f'{data_path}tests/overall_metrics_dict-10'
    p12 = f'{data_path}tests/overall_metrics_dict-12'
    p13 = f'{data_path}tests/overall_metrics_dict-13'
    p20 = f'{data_path}tests/overall_metrics_dict-20'
    p21 = f'{data_path}tests/overall_metrics_dict-21'
    p23 = f'{data_path}tests/overall_metrics_dict-23'
    p30 = f'{data_path}tests/overall_metrics_dict-30'
    p31 = f'{data_path}tests/overall_metrics_dict-31'
    p32 = f'{data_path}tests/overall_metrics_dict-32'

    save_pickle(d01, p01)
    save_pickle(d02, p02)
    save_pickle(d03, p03)
    save_pickle(d10, p10)
    save_pickle(d12, p12)
    save_pickle(d13, p13)
    save_pickle(d20, p20)
    save_pickle(d21, p21)
    save_pickle(d23, p23)
    save_pickle(d30, p30)
    save_pickle(d31, p31)
    save_pickle(d32, p32)

    mutation = "Mutation_add"
    pm = get_packed_metrics([f'{data_path}tests/'], mutation)

    print(pm["Musicality"])
    print(pm["Plagiarism-dist"])

    print("================================= STYLE =================================")
    for orig, val in pm["Style"].items():
        print(orig)
        print(val)


def test_overall_evaluation():
    b = 4
    model_name = "brmf_4b_beta-96"
    mutation = "Mutation_add_sub_1"

    overall_metric_dirs = [get_eval_dir(model_name)]
    eval_path = f"{data_path}/overall_evaluation/{model_name}-{mutation}"

    do_overall_single_evaluation(overall_metric_dirs, mutation, eval_path, b)


def test_overall_evaluation_mixture_model():
    b, z = 4, 96
    model_name = "4-Lakh_Kern-96"
    mutation = "Mutation_add_0.1"

    overall_metric_dirs = [get_eval_dir(model_name)]
    eval_path = f"{data_path}/overall_evaluation/{model_name}-{mutation}"

    do_overall_single_evaluation(overall_metric_dirs, mutation, eval_path, b, z)


def test_ensamble_overall_evaluation():
    b = 4
    mutation = "Mutation_add_sub_1"
    ensamble = [m for m in models if m in mixture_models and m[0] == str(b)]
    overall_metric_dirs = [get_eval_dir(model_name) for model_name in ensamble]
    eval_path = f"{data_path}overall_evaluation/ensamble_{b}bars-{mutation}"
    do_overall_single_evaluation(overall_metric_dirs, mutation, eval_path, b)


def test_family_evaluation():
    for family_name, family in model_families.items():
        b = 4
        z = 96
        eval_paths_by_mutation = {}
        for mutation in mutations:
            eval_paths = []
            for model_name in family:
                eval_dir = get_eval_dir(model_name)
                eval_paths += [f"{eval_dir}/overall_metrics_dict-{mutation}-{s1}_to_{s2}.pkl"
                               for s1, s2 in transference_names(model_name)
                               ]
            eval_paths_by_mutation[mutation] = eval_paths
        family_eval_dir = f"{data_path}/family_evaluation/{family_name}"

        do_family_evaluation(eval_paths_by_mutation, family_eval_dir, b, z)


def test_audio_generation():
    model_name = "brmf_4b_beta-96"
    b, z = 4, 96
    s1 = "Mozart"
    s2 = "Frescobaldi"
    audios_path = get_audios_path(model_name)
    mutation = "Mutation_add_sub_0.1"

    transferred_path = get_transferred_path(s1, s2, model_name)
    eval_dir = get_eval_dir(model_name)
    transformation = f'{s1}_to_{s2}'
    successful_rolls_prefix = f"{eval_dir}/successful_rolls-{mutation}"

    audio_generation(mutation, eval_dir, transferred_path, audios_path, successful_rolls_prefix, transformation, b, z)

    assert os.path.isfile(f"{audios_path}sonata01-1_25-rec.mid")
    assert os.path.isfile(f"{audios_path}sonata01-1_25-{transformation}-{mutation}.mid")


def test_sheet_generation():
    model_name = "brmf_4b_beta-96"
    b, z = 4, 96
    s1 = "Mozart"
    s2 = "Frescobaldi"
    mutation = "Mutation_add_sub_0.1"

    transferred_path = get_transferred_path(s1, s2, model_name)
    sheets_path = get_sheets_path(model_name)
    eval_dir = get_eval_dir(model_name)
    transference = f'{s1}_to_{s2}'
    df_audios_path = f"{eval_dir}/df_audios-{transference}-{mutation}.pkl"
    df_sheets_path = f"{eval_dir}/df_sheets-{transference}-{mutation}.pkl"

    sheets_generation(sheets_path, transferred_path, transference, mutation, df_audios_path, df_sheets_path, b, z)


def test_html_generation():
    model_name = 'brmf_4b_beta-96'
    b = model_name[5]
    z = int(model_name.split("-")[-1])

    eval_dir = get_eval_dir(model_name)
    app_dir = f"{eval_dir}/app/"
    dfs_sheets_paths = []

    mutation = "Mutation_add_sub_0.1"
    s1, s2 = "Mozart", "Frescobaldi"
    transference = f'{s1}_to_{s2}'
    df_sheets_path = f"{eval_dir}/df_sheets-{transference}-{mutation}.pkl"
    dfs_sheets_paths.append(df_sheets_path)

    create_html(mutation, app_dir, dfs_sheets_paths, b, z)


def test_sample_examples():
    b, z = 4, 96
    model_name = "brmf_4b_beta-96"

    for orig in style_names(model_name):
        transferred_paths = {dest: get_transferred_path(orig, dest, model_name)
                             for dest in style_names(model_name) if orig != dest}
        output_path = get_examples_path(model_name) + f'{orig}/'

        create_examples(transferred_paths, output_path, b, z)
