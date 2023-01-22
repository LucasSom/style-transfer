import pytest

from dodo import data_analysis, preprocessed_data
from utils.files_utils import data_path

def test_task():
    b=4
    df_path = preprocessed_data(b)
    eval_dir = f"{data_path}/brmf_{b}b/Evaluation"

    data_analysis(df_path, eval_dir, b)