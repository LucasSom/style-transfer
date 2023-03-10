import pytest

from dodo import data_analysis, preprocessed_data, prepare_data
from utils.files_utils import data_path


def test_prepare_data():
    b = 4
    eval_dir = f"{data_path}/brmf_{b}b/Evaluation"
    df_path = preprocessed_data(b)

    prepare_data(df_path, eval_dir, b)


def test_closeness():
    b = 4
    eval_dir = f"{data_path}/brmf_{b}b/Evaluation/cross_val"
    df_80_indexes_path = eval_dir + '/df_80_indexes_'
    df_test_path = eval_dir + '/rolls_long_df_test_'

    data_analysis(preprocessed_data(b), df_80_indexes_path, df_test_path, eval_dir, b, 'closeness', cv=True)


def test_musicality():
    b = 4
    eval_dir = f"{data_path}/brmf_{b}b/Evaluation/cross_val"
    df_80_indexes_path = eval_dir + '/df_80_indexes_'
    df_test_path = eval_dir + '/rolls_long_df_test_'

    data_analysis(preprocessed_data(b), df_80_indexes_path, df_test_path, eval_dir, b, 'musicality', cv=True)


def test_style_histograms():
    b = 4
    eval_dir = f"{data_path}/brmf_{b}b/Evaluation"
    df_80_indexes_path = eval_dir + '/df_80_indexes_'
    df_test_path = eval_dir + '/rolls_long_df_test_'

    data_analysis(preprocessed_data(b), df_80_indexes_path, df_test_path, eval_dir, b, 'style_histograms', cv=False)
