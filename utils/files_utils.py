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


def get_metrics_path(transferred_path: str):
    metrics_file_path = f"{transferred_path}-metrics.pkl"
    return metrics_file_path


def get_transferred_path(e_dest: str, e_orig: str, model_name: str):
    transferred_path = f"{data_path}embeddings/{model_name}/df_transferred_{e_orig}_{e_dest}.pkl"
    return transferred_path


def get_emb_path(model_name: str):
    emb_path = f"{data_path}embeddings/{model_name}/df_emb.pkl"
    return emb_path


def get_characteristics_path(model_name: str):
    characteristics_path = f"{data_path}embeddings/{model_name}/authors_characteristics.pkl"
    return characteristics_path

def get_reconstruction_path(model_name: str):
    return os.path.join(data_path, 'reconstruction', model_name,
                        'reconstruction.pkl')


def get_model_path(model_name: str):
    model_path = f"{path_saved_models + model_name}/ckpt/"
    return model_path


def get_eval_path(transferred_path: str):
    eval_path = f"{transferred_path}-eval.pkl"
    return eval_path


def get_audios_path(model_name=None, e_orig=None, e_dest=None):
    if model_name is None:
        path = os.path.join(data_path, "Audios/")
    else:
        path = os.path.join(data_path, model_name, "Audios/")

    if e_orig is None and e_dest is None:
        return path
    return os.path.join(path, f"{e_orig}_to_{e_dest}/")
