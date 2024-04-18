import os
import pickle
from typing import Union

import pandas as pd

project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_path = project_path + '/data/'
datasets_path = os.path.join(data_path, 'datasets')
datasets_debug_path = os.path.join(datasets_path, 'debug')
data_tests_path = data_path + 'tests/'
preprocessed_data_dir = data_path + 'preprocessed_data/'
original_audios_path = os.path.join(preprocessed_data_dir, 'original/audios/')


def root_file_name(p):
    return os.path.splitext(p)[0]


def file_extension(p):
    return os.path.splitext(p)[1]


def make_dirs_if_not_exists(path):
    if not os.path.isdir(path):
        os.makedirs(path)
        print("Creating directories:", path)


def save_pickle(obj: Union[pd.DataFrame, dict], file_name: str, verbose=False):
    if file_extension(file_name) != '.pkl':
        file_name += '.pkl'

    directory = os.path.dirname(file_name)
    if not os.path.isdir(directory) and directory != '':
        os.makedirs(directory)
        if verbose: print("Created directory:", directory)

    with open(file_name, 'wb') as f:
        pickle.dump(obj, f)
        if verbose: print("Saved as:", file_name)


def load_pickle(file_name: str, verbose=False):
    if file_extension(file_name) != '.pkl':
        file_name += '.pkl'

    with open(file_name, 'rb') as f:
        p = pickle.load(f)
        if verbose: print("Loaded file:", f)
        return p


def datasets_name(ds):
    composed_name = ""
    for name in ds.keys():
        composed_name += "_" + name
    return composed_name


def preprocessed_data_path(b, lmd, small=False):
    if small:
        return f"{preprocessed_data_dir}{b}-small_br.pkl"
    if lmd:
        return f"{preprocessed_data_dir}lmd_{lmd}-{b}.pkl"
    return f"{preprocessed_data_dir}bach-rag-moz-fres-{b}.pkl"


def oversample_path(model_name):
    return f"{preprocessed_data_dir}{model_name}-balanced.pkl"


def get_logs_path(model_name):
    return os.path.join(data_path, "models", model_name, 'logs/')


def get_embedding_dir(model_name):
    return os.path.join(data_path, "models", model_name, 'embeddings')


def get_embedding_path(model_name, characteristics=False):
    return os.path.join(get_embedding_dir(model_name), 'authors_characteristics' if characteristics else 'df_emb')


def get_model_paths(model_name: str):
    """Returns a tuple with model_dir, vae_dir and vae_path

    :param model_name: name of de model
    :return: model_dir, vae_dir, vae_path"""
    model_dir = os.path.join(data_path, "models", model_name)
    vae_dir = os.path.join(model_dir, "vae")

    if not os.path.isdir(model_dir):
        os.makedirs(vae_dir)

    vae_path = os.path.join(vae_dir, "saved_model.pb")
    return model_dir, vae_dir, vae_path


def get_metrics_dir(model_name: str):
    metrics_file_path = f"{data_path}models/{model_name}/metrics"
    make_dirs_if_not_exists(metrics_file_path)
    return metrics_file_path


def get_transferred_path(s1: str, s2: str, model_name: str):
    """Returns the path of the form {data_path}models/{model_name}/embeddings/df_transferred_{s1}_{s2}.pkl"""
    transferred_path = f"{data_path}models/{model_name}/embeddings/df_transferred_{s1}_{s2}.pkl"
    make_dirs_if_not_exists(os.path.dirname(transferred_path))
    return transferred_path


def get_emb_path(model_name: str):
    """Returns the path of the form {data_path}models/{model_name}/embeddings/df_emb.pkl"""
    emb_path = f"{data_path}models/{model_name}/embeddings/df_emb.pkl"
    make_dirs_if_not_exists(os.path.dirname(emb_path))
    return emb_path


def get_characteristics_path(model_name: str):
    """Returns the path of the form {data_path}models/{model_name}/embeddings/authors_characteristics.pkl"""
    characteristics_path = f"{data_path}models/{model_name}/embeddings/authors_characteristics.pkl"
    return characteristics_path


def get_reconstruction_path(model_name: str):
    """Returns the path of the form {data_path}models/{model_name}/embeddings/reconstruction.pkl"""
    return f'{data_path}models/{model_name}/embeddings/reconstruction.pkl'


def get_eval_dir(model_name: str):
    """Returns the path of the form {data_path}models/{model_name}/Evaluation"""
    eval_path = f"{data_path}models/{model_name}/Evaluation"
    make_dirs_if_not_exists(eval_path)
    return eval_path


def get_audios_path(model_name):
    """Returns the path of the form {data_path}models/{model_name}/audios/"""
    return os.path.join(data_path, "models", model_name, "audios/")


def get_sheets_path(model_name: str):
    """Returns the path of the form {data_path}models/{model_name}/sheets/"""
    path = os.path.join(data_path, "models", model_name, "sheets/")
    # path = os.path.join(path, f"{original_style}_to_{target_style}/")

    if not os.path.isdir(path):
        os.makedirs(path)
    return path


def get_examples_path(model_name: str):
    """Returns the path of the form {data_path}models/{model_name}/examples/"""
    return os.path.join(data_path, "models", model_name, "examples/")
