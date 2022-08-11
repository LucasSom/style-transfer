from doit.api import run_tasks, run
from doit.cmd_base import ModuleTaskLoader
from tensorflow.keras.models import load_model

from evaluation.app.html_maker import make_html
from evaluation.evaluation import evaluate_model
from evaluation.metrics.metrics import obtain_metrics
from model.colab_tension_vae.params import init
from model.embeddings.characteristics import calculate_characteristics
from model.embeddings.embeddings import get_reconstruction
from model.embeddings.transfer import transfer_style_to
from model.train import train_model
from preprocessing import preprocess_data
from utils.audio_management import generate_audios
from utils.files_utils import *


def preprocessed_data(b):
    return f"{preprocessed_data_path}bach-rag-moz-fres-{b}.pkl"  # TODO: Pasarlo a un archivo de configuracion


subdatasets = ["Bach", "Mozart", "Frescobaldi", "ragtime"]
styles_dict = {'b': "Bach", 'm': "Mozart", 'f': "Frescobaldi", 'r': "ragtime"}

bars = [4, 8]
models = [f"{b}-{x}{y}" for b in bars for x in 'brmf' for y in 'brmf' if x < y]

epochs = [200, 500, 1000]
checkpoints = [50, 100]

DOIT_CONFIG = {'verbosity': 2}


def preprocess(b, folders, targets):
    init(b)
    songs = {}
    for folder in folders:
        songs[folder] = [f"{folder}/{song}"
                         for song in os.listdir(f"{datasets_path}/{folder}")]
    df = preprocess_data(songs)
    save_pickle(df, targets[0])


def task_preprocess():
    """Preprocess dataset, considering the subdatasets referenced in the list at the top of this file"""
    # files = [f for sd in subdatasets for f in list(pathlib.Path(sd).glob('*.txt'))]
    for b in bars:
        yield {
            # 'file_dep': files,
            'name': f"{b}bars",
            'actions': [(preprocess, [b], {'folders': subdatasets})],
            'targets': [preprocessed_data(b)],
            'uptodate': [os.path.isfile(preprocessed_data(b))]
        }


def train(df_path, model_name):
    styles = [styles_dict[a] for a in model_name[-2:]]
    df = load_pickle(df_path)
    df = df[df['Style'].isin(styles)]
    train_model(df, model_name)


# TODO: Pasarle por parámetro a la tarea las épocas y el ckpt
def task_train():
    """Trains the model"""
    for model_name in models:
        b = model_name[0]
        init(b)
        _, model_pb_path = get_model_path(model_name)
        yield {
            'name': f"{model_name}",
            'file_dep': [preprocessed_data(b)],
            'actions': [(train, [preprocessed_data(b), model_name])],
            'targets': [model_pb_path]
        }


def analyze_training(df_path, model_path, targets):
    model = load_model(model_path)
    model_name = os.path.basename(model_path)
    df = load_pickle(df_path)
    df_reconstructed = get_reconstruction(df, model, model_name, inplace=False)
    save_pickle(df_reconstructed, targets[0])


def task_test():
    """Shows the reconstruction of the model over an original song"""
    for model_name in models:
        b = model_name[0]
        init(b)
        model_path, model_pb_path = get_model_path(model_name)
        yield {
            'name': f"{model_name}",
            'file_dep': [preprocessed_data(b), model_pb_path],
            'actions': [(analyze_training, [preprocessed_data(b), model_path])],
            'targets': [get_reconstruction_path(model_name)]
        }


def do_embeddings(df_path, model_path, characteristics_path, emb_path):
    model = load_model(model_path)
    df = load_pickle(df_path)

    df_emb, authors_char = calculate_characteristics(df, model)

    save_pickle(authors_char, characteristics_path)
    save_pickle(df_emb, emb_path)


def task_embeddings():
    """Calculate the embeddings for each author/style and song"""
    for model_name in models:
        b = model_name[0]
        init(b)
        model_path, model_pb_path = get_model_path(model_name)
        characteristics_path = get_characteristics_path(model_name)
        emb_path = get_emb_path(model_name)

        yield {
            'name': f"{model_name}",
            'file_dep': [preprocessed_data(b), model_pb_path],
            'actions': [(do_embeddings,
                         [preprocessed_data(b), os.path.dirname(model_path), characteristics_path, emb_path]
                         )],
            'targets': [characteristics_path, emb_path],
            'uptodate': [os.path.isfile(characteristics_path) and os.path.isfile(emb_path)]
        }


def do_transfer(df_emb, model_path, characteristics, orig, target, transferred_path):
    df_emb = load_pickle(df_emb)
    model = load_model(model_path)
    characteristics = load_pickle(characteristics)
    model_name = os.path.basename(model_path)

    df_transferred = transfer_style_to(df_emb, model, model_name, characteristics, original_style=orig,
                                       target_style=target)
    save_pickle(df_transferred, transferred_path)


def task_transfer_style():
    """Do the transference of style from a roll to another style"""
    for model_name in models:

        model_path, model_pb_path = get_model_path(model_name)
        characteristics_path = get_characteristics_path(model_name)
        emb_path = get_emb_path(model_name)

        for e_orig, e_dest in [(model_name[-2], model_name[-1]), (model_name[-1], model_name[-2])]:
            transferred_path = get_transferred_path(e_orig, e_dest, model_name)
            yield {
                'name': f"{model_name}-{e_orig}_to_{e_dest}",
                'file_dep': [emb_path, model_pb_path, characteristics_path],
                'actions': [(do_transfer,
                             [emb_path, os.path.dirname(model_path),
                              characteristics_path, e_orig, e_dest, transferred_path]
                             )],
                'targets': [transferred_path],
                'verbosity': 2
            }


def audio_generation(transferred_path, audios_path, suffix=None, column=None, orig=None, dest=None):
    df_transferred = load_pickle(transferred_path)
    generate_audios(df_transferred, audios_path, column=column, suffix=suffix, verbose=1)
    make_html(df_transferred, orig=orig, targets=[dest], audios_path=audios_path)
    # TODO: Que targets solo tome al target.


def task_sample_audios():
    """Produce the midis generated by the style transfer"""
    for model_name in models:
        recon_path = get_reconstruction_path(model_name)
        audios_path = get_audios_path(model_name)
        yield {
            'name': f'{model_name}-orig',
            'file_dep': [recon_path],
            'actions': [(audio_generation, (recon_path, audios_path, 'orig', 'roll'))],
            'uptodate': [False]
        }
        yield {
            'name': f'{model_name}-reconstruction',
            'file_dep': [recon_path],
            'actions': [(audio_generation, (recon_path, audios_path, 'recon', 'Embedding-NewRoll'))],
            'uptodate': [False]
        }
        for e_orig, e_dest in [(model_name[-2], model_name[-1]), (model_name[-1], model_name[-2])]:
            transferred_path = get_transferred_path(e_orig, e_dest, model_name)
            suffix = f'{e_orig}_to_{e_dest}'
            yield {
                'name': f"{model_name}-{e_orig}_to_{e_dest}",
                'file_dep': [transferred_path, recon_path],
                'actions': [(audio_generation,
                             [transferred_path, audios_path],
                             dict(suffix=suffix, orig=e_orig,
                                  dest=e_dest)
                             )],
                'verbosity': 2,
                'uptodate': [False]
            }


def generate_sheets(transferred_path, sheets_path, suffix=None, column=None):
    df_transferred = load_pickle(transferred_path)
    column = column if column is not None else df_transferred.columns[-1]

    titles = (df_transferred['Title'] if suffix is None
              else df_transferred['Title'].map(lambda t: f'{t}_{suffix}'))
    rolls = df_transferred[column]
    for title, roll in zip(titles, rolls):
        sheet_path = os.path.join(sheets_path, title)
        roll.display_score(file_name=sheet_path, fmt='png', do_display=False)


def task_sample_sheets():
    """Produce the sheets generated by the style transference"""
    for model_name in models:
        recon_path = get_reconstruction_path(model_name)
        sheets_path = get_sheets_path(model_name, orig=True)
        yield {
            'name': f'{model_name}-orig',
            'file_dep': [recon_path],
            'actions': [(generate_sheets, (recon_path, sheets_path, 'orig', 'roll'))],
            'uptodate': [False]
        }
        sheets_path = get_sheets_path(model_name, orig=False)
        yield {
            'name': f'{model_name}-reconstruction',
            'file_dep': [recon_path],
            'actions': [(generate_sheets, (recon_path, sheets_path, 'recon'))],
            'uptodate': [False]
        }
        for e_orig, e_dest in [(model_name[-2], model_name[-1]), (model_name[-1], model_name[-2])]:
            transferred_path = get_transferred_path(e_orig, e_dest, model_name)
            sheets_path = get_sheets_path(model_name, original_style=e_orig, target_style=e_dest)
            suffix = f'{e_orig}_to_{e_dest}'
            # TODO: En realidad es irrelevante el estilo de origen en el nombre de la canción. Podría omitirse
            yield {
                'name': f"{model_name}-{e_orig}_to_{e_dest}",
                'file_dep': [transferred_path, recon_path],
                'actions': [(generate_sheets, [transferred_path, sheets_path, suffix])],
                'verbosity': 2,
                'uptodate': [False]
            }


def calculate_metrics(trans_path, e_orig, e_dest, metrics_file_path):
    df_transferred = load_pickle(trans_path)
    metrics = obtain_metrics(df_transferred, e_orig, e_dest, None)
    save_pickle(metrics, metrics_file_path)


def task_metrics():
    """Calculate different metrics for a produced dataset"""
    for model_name in models:
        for e_orig, e_dest in [(model_name[-2], model_name[-1]), (model_name[-1], model_name[-2])]:
            transferred_path = get_transferred_path(e_orig, e_dest, model_name)
            metrics_path = get_metrics_path(transferred_path)
            yield {
                'name': f"{model_name}-{e_orig}_to_{e_dest}",
                'file_dep': [transferred_path],
                'actions': [(calculate_metrics, [transferred_path, e_orig, e_dest, metrics_path])],
                'targets': [metrics_path],
                'verbosity': 2
            }


def do_evaluation(trans_path, eval_path):
    df_transferred = load_pickle(trans_path)
    metrics = load_pickle(get_metrics_path(trans_path))
    evaluation_results = evaluate_model(df_transferred, metrics)
    save_pickle(evaluation_results, eval_path)


def task_evaluation():
    """Evaluate the model considering the calculated metrics"""
    for model_name in models:
        for style1, style2 in [(model_name[-2], model_name[-1]), (model_name[-1], model_name[-2])]:

            transferred_path = get_transferred_path(style1, style2, model_name)
            eval_path = get_eval_path(transferred_path)
            yield {
                'name': f"{model_name}-{style1}_to_{style2}",
                'file_dep': [transferred_path, get_transferred_path(style1, style2, transferred_path)],
                'actions': [(do_evaluation, [transferred_path])],
                'targets': [eval_path],
                'verbosity': 2
            }


if __name__ == '__main__':
    import doit
    g = globals()
    # run_tasks(ModuleTaskLoader(g), {'train:4-br':1})
    run(g)
