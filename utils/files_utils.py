import os
import pickle
from typing import Union

import pandas as pd

project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_path = project_path + '/data/'
datasets_path = os.path.join(data_path, 'datasets')
datasets_debug_path = os.path.join(datasets_path, 'debug')
data_tests_path = data_path + 'tests/'
preprocessed_data_path = data_path + 'preprocessed_data/'
path_saved_models = data_path + 'saved_models/'
original_audios_path = os.path.join(preprocessed_data_path, 'original/audios/')


def get_logs_path(model_name):
    return os.path.join(data_path, model_name, 'logs/')


def get_embedding_dir(model_name):
    return os.path.join(data_path, model_name, 'embeddings')


def get_embedding_path(model_name, characteristics=False):
    return os.path.join(get_embedding_dir(model_name), 'authors_characteristics' if characteristics else 'df_emb')


def get_model_paths(model_name: str):
    model_dir = os.path.join(data_path, model_name)
    vae_dir = os.path.join(model_dir, "vae")

    if not os.path.isdir(model_dir):
        os.makedirs(vae_dir)
    # logs_dir = os.path.join(model_dir, "logs")
    vae_path = os.path.join(vae_dir, "saved_model.pb")
    return model_dir, vae_dir, vae_path


def root_file_name(p):
    return os.path.splitext(p)[0]


def file_extension(p):
    return os.path.splitext(p)[1]


def datasets_name(ds):
    composed_name = ""
    for name in ds.keys():
        composed_name += "_" + name
    return composed_name


def save_pickle(obj: Union[pd.DataFrame, dict], file_name: str, verbose=False):
    if file_extension(file_name) == '':
        file_name += '.pkl'

    directory = os.path.dirname(file_name)
    if not os.path.isdir(directory) and directory != '':
        os.makedirs(directory)
        if verbose: print("Created directory:", directory)

    with open(file_name, 'wb') as f:
        pickle.dump(obj, f)
        if verbose: print("Saved as:", file_name)


def load_pickle(file_name: str, verbose=False):
    if file_extension(file_name) == '':
        file_name += '.pkl'

    with open(file_name, 'rb') as f:
        p = pickle.load(f)
        if verbose: print("Loaded file:", f)
        return p


# def get_preproc_small_path(b):
#     return os.path.join(data_path, 'preprocessed_data',
#                         f'bach-rag-moz-fres-{b}_small.pkl')


def get_metrics_path(transferred_path: str):
    metrics_file_path = f"{transferred_path}-metrics.pkl"
    return metrics_file_path


def get_transferred_path(e_orig: str, e_dest: str, model_name: str):
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
    model_pb_path = os.path.join(model_path, 'saved_model.pb')
    return model_path, model_pb_path


def get_eval_path(transferred_path: str):
    eval_path = f"{transferred_path}-eval.pkl"
    return eval_path


def get_audios_path(model_name=None, reconstruction=False, e_orig=None, e_dest=None):
    if model_name is None:  # ie, original
        return os.path.join(data_path, "original/audios")
        # return original_audios_path
    path = os.path.join(data_path, model_name, "audios/")

    if e_orig is not None or e_dest is not None:
        return os.path.join(path, f"{e_orig}_to_{e_dest}/")

    if reconstruction:
        return os.path.join(path, "reconstruction/audios")
    return path


def get_sheets_path(model_name: str = None, original_style: str = None, target_style: str = None, orig=False):
    """
    :param model_name: name of the containing folder inside the data directory.
    :param original_style: name of style of the original song.
    :param target_style: name of transferred style.
    :param orig: if original_style and target_style are None, determines the suffix between 'orig' and 'recon'.
    """
    if model_name is None:
        path = os.path.join(data_path, "original/sheets/")
    else:
        path = os.path.join(data_path, model_name, "sheets/")

    if original_style is None and target_style is None:
        path = os.path.join(path, f"{'orig' if orig else 'recon'}/")
    else:
        path = os.path.join(path, f"{original_style}_to_{target_style}/")

    if not os.path.isdir(path):
        os.makedirs(path)
    return path
