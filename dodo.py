import os

from tensorflow.keras.models import load_model

from evaluation.evaluation import evaluate_model
from evaluation.metrics.metrics import obtain_metrics
from model.colab_tension_vae.params import init
from model.embeddings.characteristics import calculate_characteristics
from model.embeddings.transfer import transfer_style_to
from pipeline_tests.test_training import test_reconstruction
from model.train import train_model
from preprocessing import preprocess_data
from utils.files_utils import save_pickle, data_path, path_saved_models, load_pickle, preprocessed_data_path


def preprocessed_data(b):
    return f"{preprocessed_data_path}bach_rag_moz_fres-{b}.pkl"  # TODO: Pasarlo a un archivo de configuracion


subdatasets = ["Bach", "Mozart", "Frescobaldi", "ragtime"]

models = {4: 'brmf_4b', 8: 'brmf_8b'}
epochs = [200, 500, 1000]
checkpoints = [50, 100]

DOIT_CONFIG = {'verbosity': 2}


def preprocess(b, folders, targets):
    init(b)
    songs = {}
    for folder in folders:
        songs[folder] = [f"{folder}/{song}" for song in os.listdir(data_path + folder)]
    df = preprocess_data(songs)
    save_pickle(df, targets[0])


def task_preprocess():
    """Preprocess dataset, considering the subdatasets referenced in the list at the top of this file"""
    # files = [f for sd in subdatasets for f in list(pathlib.Path(sd).glob('*.txt'))]
    for b in models:
        yield {
            # 'file_dep': files,
            'name': f"{b}bars",
            'actions': [(preprocess, [b], {'folders': subdatasets})],
            'targets': [preprocessed_data(b)],
            'uptodate': [os.path.isfile(preprocessed_data(b))]
        }


# TODO: Pasarle por parámetro a la tarea las épocas y el ckpt
def task_train():
    """Trains the model"""
    for b, model_name in models.items():
        init(b)
        for e in epochs:
            for c in checkpoints:
                # path_to_save = f"{path_saved_models + model_name}/ckpt/saved_model.pb"
                yield {
                    'name': f"{model_name}-e{e}-ckpt{c}",
                    'file_dep': [preprocessed_data(b)],
                    'actions': [(train_model, [preprocessed_data(b), model_name, e, c])],
                    # 'targets': [path_to_save],
                }


def analyze_training(df, model_path, model_name):
    model = load_model(model_path)
    test_reconstruction(df, model, model_name)


def task_test():
    """Shows the reconstruction of the model over an original song"""
    for b, model_name in models.items():
        init(b)
        for e in epochs:
            model_path = f"{path_saved_models + model_name}/ckpt/saved_model.pb"
            yield {
                'name': f"{model_name}-e{e}",
                'file_dep': [preprocessed_data(b), model_path],
                'actions': [(analyze_training, [preprocessed_data(b), model_path, model_name])],
            }


def do_embeddings(df, model_path, characteristics_path, emb_path):
    model = load_model(model_path)

    df_emb, authors_char = calculate_characteristics(df, model)

    save_pickle(authors_char, characteristics_path)
    save_pickle(df_emb, emb_path)


def task_embeddings():
    """Calculate the embeddings for each author/style and song"""
    for b, model_name in models.items():
        init(b)
        model_path = f"{path_saved_models + model_name}/ckpt/saved_model.pb"
        characteristics_path = f"{data_path}embeddings/{model_name}/authors_characteristics.pkl"
        emb_path = f"{data_path}embeddings/{model_name}/df_emb.pkl"

        yield {
            'name': f"{model_name}",
            'file_dep': [preprocessed_data(b), model_path],
            'actions': [(do_embeddings, [preprocessed_data(b), model_path, characteristics_path, emb_path])],
            'targets': [characteristics_path, emb_path]
        }


def do_transfer(df_emb, model_path, characteristics, orig, target, transferred_path):
    df_emb = load_pickle(df_emb)
    model = load_model(model_path)
    characteristics = load_pickle(characteristics)

    df_transfered = transfer_style_to(df_emb, model, characteristics, original_style=orig, target_style=target)
    save_pickle(df_transfered, f"{transferred_path}_{orig}_{target}.pkl")


def task_transfer_style():
    """Do the transference of style from a roll to another style"""
    for model_name in models.values():

        model_path = f"{path_saved_models + model_name}/ckpt/saved_model.pb"
        characteristics_path = f"{data_path}embeddings/{model_name}/authors_characteristics.pkl"
        emb_path = f"{data_path}embeddings/{model_name}/df_emb.pkl"

        for e_orig in subdatasets:
            for e_dest in subdatasets:
                if e_orig != e_dest:
                    transferred_path = f"{data_path}embeddings/{model_name}/df_transferred_{e_orig}_{e_dest}.pkl"
                    yield {
                        'name': f"{model_name}-{e_orig}_to_{e_dest}",
                        'file_dep': [emb_path, model_path, characteristics_path],
                        'actions': [(do_transfer,
                                     [emb_path, model_path, characteristics_path, e_orig, e_dest, transferred_path])],
                        'targets': [transferred_path],
                        'verbosity': 2
                    }


def calculate_metrics(trans_path, e_orig, e_dest):
    df_transferred = load_pickle(trans_path)
    metrics = obtain_metrics(df_transferred, e_orig, e_dest)
    save_pickle(metrics, f"{trans_path}-metrics.pkl")


def task_metrics():
    """Calculate different metrics for a produced dataset"""
    for model_name in models.values():
        for e_orig in subdatasets:
            for e_dest in subdatasets:
                if e_orig != e_dest:
                    transferred_path = f"{data_path}embeddings/{model_name}/df_transferred_{e_orig}_{e_dest}.pkl"
                    yield {
                        'name': f"{model_name}-{e_orig}_to_{e_dest}",
                        'file_dep': [transferred_path],
                        'actions': [(calculate_metrics, [transferred_path, e_orig, e_dest])],
                        'targets': [f"{transferred_path}-metrics.pkl"],
                        'verbosity': 2
                    }


def do_evaluation(trans_path):
    df_transferred = load_pickle(trans_path)
    metrics = load_pickle(f"{trans_path}-metrics.pkl")
    evaluation_results = evaluate_model(df_transferred, metrics)
    save_pickle(evaluation_results, f"{trans_path}-eval.pkl")


def task_evaluation():
    """Evaluate the model considering the calculated metrics"""
    for model_name in models.values():
        for e_orig in subdatasets:
            for e_dest in subdatasets:
                if e_orig != e_dest:
                    transferred_path = f"{data_path}embeddings/{model_name}/df_transferred_{e_orig}_{e_dest}.pkl"
                    yield {
                        'name': f"{model_name}-{e_orig}_to_{e_dest}",
                        'file_dep': [transferred_path, f"{transferred_path}-metrics.pkl"],
                        'actions': [(do_evaluation, [transferred_path])],
                        'targets': [f"{transferred_path}-eval.pkl"],
                        'verbosity': 2
                    }
