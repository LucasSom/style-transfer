import os
import pickle
from typing import Union

import pandas as pd

project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_path = project_path + '/data/'
data_tests_path = data_path + 'tests/'
preprocessed_data_path = data_path + 'preprocessed_data/'
path_saved_models = data_path + 'saved_models/'
logs_path = data_path + 'logs/'


def datasets_name(ds):
    composed_name = ""
    for name in ds.keys():
        composed_name += "_" + name
    return composed_name


def save_pickle(obj: Union[pd.DataFrame, dict], file_name: str, verbose=False):
    if os.path.splitext(file_name)[1] == '':
        file_name += '.pkl'

    dir = os.path.dirname(file_name)
    if not os.path.isdir(dir) and dir != '':
        os.makedirs(dir)
        if verbose: print("Created directory:", dir)

    with open(file_name, 'wb') as f:
        pickle.dump(obj, f)
        if verbose: print("Saved as:", file_name)


def load_pickle(file_name: str, verbose=False):
    if os.path.splitext(file_name)[1] == '':
        file_name += '.pkl'

    with open(file_name, 'rb') as f:
        p = pickle.load(f)
        if verbose: print("Loaded file:", f)
        return p
