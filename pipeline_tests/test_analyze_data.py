import pytest

from dodo import data_analysis, preprocessed_data
from utils.files_utils import data_path


def test_plot_styles_bigrams_entropy():
    ...


def test_task():
    b=4
    eval_dir = f"{data_path}/brmf_{b}b/Evaluation"
    df_80_path = eval_dir + '/df_80.pkl'
    df_test_path = eval_dir + '/rolls_long_df_test.pkl'

    data_analysis(df_80_path, df_test_path, eval_dir, b)