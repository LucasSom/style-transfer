import os.path
from copy import copy

from doit.api import run
from keras.saving.save import load_model

from evaluation.app.html_maker import make_html
from evaluation.evaluation import evaluate_model
from evaluation.metrics.metrics import obtain_metrics, styles_bigrams_entropy
from model.colab_tension_vae.params import init
from model.embeddings.characteristics import obtain_characteristics
from model.embeddings.embeddings import get_reconstruction, obtain_embeddings
from model.embeddings.transfer import transfer_style_to
from model.train import train_model
from preprocessing import preprocess_data
from utils.audio_management import generate_audios
from utils.files_utils import *
from utils.plots_utils import calculate_TSNEs, plot_tsne, plot_tsnes_comparison, plot_embeddings, \
    plot_characteristics_distributions, plot_styles_bigrams_entropy, plot_styles_heatmaps
from utils.utils import show_sheets, sample_uniformly

subdatasets = ["Bach", "Mozart", "Frescobaldi", "ragtime"]
small_subdatasets = ["small_Bach", "small_ragtime"]
styles_dict = {'b': "Bach", 'm': "Mozart", 'f': "Frescobaldi", 'r': "ragtime"}

bars = [4]  # [4, 8]
old_models = ['brmf_4b', 'brmf_8b']
models = [f"{b}-{x}{y}" for b in bars for x in 'brmf' for y in 'brmf' if x < y] + old_models + ["4-small_br"]
# models = ["brmf_4b"]


epochs = [200, 500, 1000]
checkpoints = [50, 100]


def styles_names(model_name):
    if len(model_name) == 4:
        m1 = styles_dict[model_name[-1]]
        m2 = styles_dict[model_name[-2]]
        styles = [(m1, m2), (m2, m1)]
    elif "small" in model_name:
        b, r = small_subdatasets
        styles = [(b, r), (r, b)]
    else:
        b, f, m, r = subdatasets
        styles = [(b, f), (b, m), (b, r), (f, m), (f, r), (m, r)]
        for s1, s2 in copy(styles):
            styles.append((s2, s1))
    return styles


def preprocessed_data(b, small=False):
    if small:
        return f"{preprocessed_data_path}{b}-small_br.pkl"
    return f"{preprocessed_data_path}bach-rag-moz-fres-{b}.pkl"  # TODO: Pasarlo a un archivo de configuracion


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
    yield {
        # 'file_dep': files,
        'name': f"small_4bars",
        'actions': [(preprocess, [4], {'folders': small_subdatasets})],
        'targets': [preprocessed_data(4, True)],
        'uptodate': [os.path.isfile(preprocessed_data(4, True))]
    }


def data_analysis(df_path, eval_dir, b):
    init(b)
    df = load_pickle(df_path)

    entropies = styles_bigrams_entropy(df)
    plot_styles_bigrams_entropy(entropies, eval_dir)

    plot_styles_heatmaps(df, eval_dir)


def task_analyze_data():
    """Get an analysis of the dataset"""
    for b in bars:
        eval_dir = f"{data_path}/brmf_{b}b/Evaluation"

        yield {
            'name': f"{b}bars",
            'file_dep': [preprocessed_data(b)],
            'actions': [(data_analysis, [preprocessed_data(b), eval_dir, b])],
            'targets': [eval_dir + '/plots/styles_complexity.png'],
            'uptodate': [True]
        }


def train(df_path, model_name, b):
    init(b)
    if_train = input("Do you want to train the model [Y/n]? ")
    if if_train in ['Y', 'y', 'S', 's']:
        styles = ["small_Bach", "small_ragtime"] if "small" in model_name else [styles_dict[a] for a in model_name[2:4]]
        df = load_pickle(df_path)
        df = df[df['Style'].isin(styles)]
        train_model(df, model_name)
    elif if_train in ['N', 'n']:
        print("Skipping training")
    else:
        print("Try again")


def task_train():
    """Trains the model"""
    for model_name in models:
        b = model_name[-2] if model_name in old_models else model_name[0]
        small = "small" in model_name
        vae_path = get_model_paths(model_name)[2]
        yield {
            'name': f"{model_name}",
            'file_dep': [preprocessed_data(b, small)],
            'actions': [(train, [preprocessed_data(b, small), model_name, b])],
            'targets': [vae_path],
            # 'uptodate': [True]
        }


def analyze_training(df_path, model_name, b, targets):
    init(b)
    model_path, vae_dir, _ = get_model_paths(model_name)
    model = load_model(vae_dir)
    model_name = os.path.basename(model_name)
    plots_path = os.path.join(data_path, model_path, "plots")
    df = load_pickle(df_path)

    df_emb = obtain_embeddings(df, model, inplace=True)
    tsne_emb = calculate_TSNEs(df_emb, column_discriminator="Style")[0]

    plot_tsnes_comparison(df_emb, tsne_emb, plots_path)
    plot_tsne(df_emb, tsne_emb, plots_path)

    df_reconstructed = get_reconstruction(df, model, model_name, 500, inplace=False)
    save_pickle(df_reconstructed, targets[0])


def task_test():
    """Shows the reconstruction of the model over an original song and a t-SNE plot of the songs in the latent space."""
    for model_name in models:
        b = model_name[-2] if model_name in old_models else model_name[0]
        small = "small" in model_name
        vae_path = get_model_paths(model_name)[2]
        yield {
            'name': f"{model_name}",
            'file_dep': [preprocessed_data(b, small), vae_path],
            'actions': [(analyze_training, [preprocessed_data(b, small), model_name, b])],
            'targets': [get_reconstruction_path(model_name)]
        }


def do_embeddings(df_path, model_path, vae_path, characteristics_path, emb_path, b):
    init(b)
    print(os.path.abspath(vae_path))
    model = load_model(os.path.abspath(vae_path))
    plots_dir = os.path.join(model_path, "plots")
    df = load_pickle(df_path)

    df_emb, styles_char = obtain_characteristics(df, model)

    plot_embeddings(df_emb, "Embedding", {n: s.embedding for n, s in styles_char.items()}, plots_dir,
                    include_songs=True)
    plot_characteristics_distributions(styles_char, plots_dir, "Distributions_characteristics")

    save_pickle(styles_char, characteristics_path)
    save_pickle(df_emb, emb_path)


def task_embeddings():
    """Calculate the embeddings for each author/style and song"""
    for model_name in models:
        b = model_name[-2] if model_name in old_models else model_name[0]
        model_path, vae_dir, vae_path = get_model_paths(model_name)
        characteristics_path = get_characteristics_path(model_name)
        emb_path = get_emb_path(model_name)
        small = "small" in model_name

        yield {
            'name': f"{model_name}",
            'file_dep': [preprocessed_data(b, small), vae_path],
            'actions': [(do_embeddings,
                         [preprocessed_data(b, small),
                          model_path,
                          vae_dir,
                          characteristics_path,
                          emb_path, b]
                         )],
            'targets': [characteristics_path, emb_path],
            'uptodate': [os.path.isfile(characteristics_path) and os.path.isfile(emb_path)]
            # 'uptodate': [False]
        }


def do_transfer(df_emb, model_path, characteristics, transferred_path, s1, s2, b=4):
    init(b)
    df_emb = load_pickle(df_emb)
    model = load_model(model_path)
    characteristics = load_pickle(characteristics)
    model_name = os.path.basename(model_path)

    df_transferred = pd.concat([
        transfer_style_to(df_emb, model, model_name, characteristics, original_style=s1, target_style=s2),
        transfer_style_to(df_emb, model, model_name, characteristics, original_style=s2, target_style=s1),
    ])

    save_pickle(df_transferred, transferred_path)


def task_transfer_style():
    """Do the transference of style from a roll to another style"""
    for model_name in models:
        b = model_name[-2] if model_name in old_models else model_name[0]
        model_path, vae_dir, vae_path = get_model_paths(model_name)
        characteristics_path = get_characteristics_path(model_name)
        emb_path = get_emb_path(model_name)

        s1, s2 = styles_names(model_name)[0]
        transferred_path = get_transferred_path(s1, s2, model_name)
        yield {
            'name': model_name,
            'file_dep': [emb_path, vae_path, characteristics_path],
            'actions': [(do_transfer,
                         [emb_path, vae_dir, characteristics_path, transferred_path, s1, s2, b]
                         )],
            'targets': [transferred_path],
            'verbosity': 2,
            # 'uptodate': [False]
        }


def calculate_metrics(trans_path, char_path, metrics_dir, model_name, b=4):
    init(b)
    s1, s2 = styles_names(model_name)[0]
    df_transferred = load_pickle(trans_path)
    styles = load_pickle(char_path)

    metrics1 = obtain_metrics(df_transferred, s1, s2, styles, 'plagiarism', 'intervals', 'rhythmic_bigrams',
                              'musicality')
    save_pickle(metrics1, f"{metrics_dir}/metrics_{s1}_to_{s2}")

    metrics2 = obtain_metrics(df_transferred, s2, s1, styles, 'intervals', 'rhythmic_bigrams', 'musicality')
    metrics2['plagiarism'] = metrics1['plagiarism']
    save_pickle(metrics2, f"{metrics_dir}/metrics_{s2}_to_{s1}")


def task_metrics():
    """Calculate different metrics for a produced dataset"""
    for model_name in models:
        b = model_name[-2] if model_name in old_models else model_name[0]
        s1, s2 = styles_names(model_name)[0]
        transferred_path = get_transferred_path(s1, s2, model_name)
        characteristics_path = get_characteristics_path(model_name)
        metrics_path = get_metrics_dir(transferred_path)
        yield {
            'name': model_name,
            'file_dep': [transferred_path, characteristics_path],
            'actions': [(calculate_metrics, [transferred_path, characteristics_path, metrics_path, model_name, b])],
            'targets': [f"{metrics_path}/metrics_{s1}_to_{s2}.pkl", f"{metrics_path}/metrics_{s2}_to_{s1}.pkl"],
            'verbosity': 2,
            # 'uptodate': [False]
        }


def do_evaluation(trans_path, styles_path, eval_dir, s1, s2, b=4):
    init(b)
    metrics_dir = get_metrics_dir(trans_path)

    df_transferred = load_pickle(trans_path)
    styles = load_pickle(styles_path)

    metrics = load_pickle(f"{metrics_dir}/metrics_{s1}_to_{s2}")
    successful_rolls, table = evaluate_model(df_transferred, metrics, styles, eval_path=eval_dir)
    save_pickle(successful_rolls, f"{eval_dir}/successful_rolls-{s1}_to_{s2}")
    save_pickle(table, f"{eval_dir}/results-{s1}_to_{s2}")
    for t in table.values():
        print(t)

    metrics = load_pickle(f"{metrics_dir}/metrics_{s2}_to_{s1}")
    successful_rolls, table = evaluate_model(df_transferred, metrics, styles, eval_path=eval_dir)
    save_pickle(successful_rolls, f"{eval_dir}/successful_rolls-{s2}_to_{s1}")
    save_pickle(table, f"{eval_dir}/results-{s2}_to_{s1}")
    for t in table.values():
        print(t)


def task_evaluation():
    """Evaluate the model considering the calculated metrics"""
    for model_name in models:
        b = model_name[-2] if model_name in old_models else model_name[0]
        s1, s2 = styles_names(model_name)[0]

        transferred_path = get_transferred_path(s1, s2, model_name)
        styles_path = get_characteristics_path(model_name)
        metrics_dir = get_metrics_dir(transferred_path)
        eval_dir = get_eval_dir(transferred_path)
        yield {
            'name': model_name,
            'file_dep': [transferred_path, styles_path,
                         f"{metrics_dir}/metrics_{s1}_to_{s2}.pkl", f"{metrics_dir}/metrics_{s2}_to_{s1}.pkl"],
            'actions': [(do_evaluation, [transferred_path, styles_path, eval_dir, s1, s2, b])],
            'targets': [f"{eval_dir}/successful_rolls-{s1}_to_{s2}.pkl",
                        f"{eval_dir}/successful_rolls-{s2}_to_{s1}.pkl"],
            'verbosity': 2,
            # 'uptodate': [False]
        }


def audio_generation(transferred_path, audios_path, succ_rolls_prefix=None,
                     suffix=None, orig=None, dest=None, b=4):
    init(b)
    if succ_rolls_prefix is None:
        df_transferred = load_pickle(transferred_path)
        generate_audios(df_transferred, audios_path, suffix=suffix, verbose=1)
        make_html(df_transferred, orig=orig, targets=[dest], app_dir=audios_path)
        # TODO: Que targets solo tome al target.
    else:
        successful_dfs = load_pickle(f"{succ_rolls_prefix}{suffix}")
        df_html = pd.DataFrame()
        for k, df in successful_dfs.items():
            df = sample_uniformly(df, f"{k} rank", n=5)
            original_files, new_files = generate_audios(df, audios_path, f"{k}-{suffix}", 1)
            df["Original audio files"] = original_files
            df["New audio files"] = new_files
            df_html = pd.concat([df_html, df])

        df = load_pickle(transferred_path)
        df = df.sample(n=5, random_state=42)

        original_files, new_files = generate_audios(df, audios_path, f"random-{suffix}", 1)
        df["Original audio files"] = original_files
        df["New audio files"] = new_files
        df_html = pd.concat([df_html, df])

        make_html(df_html, orig=orig, targets=[dest], app_dir=os.path.dirname(os.path.dirname(audios_path)) + '/app/')


def task_sample_audios():
    """Produce the midis generated by the style transfer"""
    for model_name in models:
        recon_path = get_reconstruction_path(model_name)
        audios_path = get_audios_path(model_name)

        yield {
            'name': f'{model_name}-orig',
            'file_dep': [recon_path],
            'actions': [(audio_generation, [recon_path, audios_path], dict(suffix="orig", column="roll"))],
            # 'uptodate': [False]
        }
        yield {
            'name': f'{model_name}-reconstruction',
            'file_dep': [recon_path],
            'actions': [(audio_generation, [recon_path, audios_path], dict(suffix='recon', column='NewRoll'))],
            # 'uptodate': [False]
        }
        for s1, s2 in styles_names(model_name):
            transferred_path = get_transferred_path(s1, s2, model_name)
            eval_dir = get_eval_dir(transferred_path)
            suffix = f'{s1}_to_{s2}'
            successful_rolls_prefix = f"{eval_dir}/successful_rolls-"
            yield {
                'name': f"{model_name}-{suffix}",
                'file_dep': [f"{successful_rolls_prefix}{suffix}.pkl", recon_path],
                'actions': [(audio_generation,
                             [transferred_path, audios_path, successful_rolls_prefix],
                             dict(suffix=suffix, orig=s1, dest=s2)
                             )],
                'verbosity': 2,
                'uptodate': [False]
            }


def sheets_generation(transferred_path, sheets_path, suffix=None, column=None, succ_rolls_prefix=None):
    if succ_rolls_prefix is None:
        df_transferred = load_pickle(transferred_path)
        column = column if column is not None else df_transferred.columns[-1]

        show_sheets(df_transferred, column, sheets_path, suffix)
    else:
        df_successful = load_pickle(f"{succ_rolls_prefix}{suffix}")
        show_sheets(df_successful, "NewRoll", sheets_path, suffix)


def task_sample_sheets():
    """Produce the sheets generated by the style transference"""
    for model_name in models:
        recon_path = get_reconstruction_path(model_name)
        sheets_path = get_sheets_path(model_name, orig=True)
        yield {
            'name': f'{model_name}-orig',
            'file_dep': [recon_path],
            'actions': [(sheets_generation, (recon_path, sheets_path, 'orig', 'roll'))],
            'uptodate': [False]
        }
        sheets_path = get_sheets_path(model_name, orig=False)
        yield {
            'name': f'{model_name}-reconstruction',
            'file_dep': [recon_path],
            'actions': [(sheets_generation, (recon_path, sheets_path, 'recon'))],
            'uptodate': [False]
        }

        for s1, s2 in styles_names(model_name):
            transferred_path = get_transferred_path(s1, s2, model_name)
            sheets_path = get_sheets_path(model_name, original_style=s1, target_style=s2)
            eval_dir = get_eval_dir(transferred_path)
            suffix = f'{s1}_to_{s2}'
            successful_rolls_prefix = f"{eval_dir}/successful_rolls-"
            yield {
                'name': f"{model_name}-{suffix}",
                'file_dep': [f"{successful_rolls_prefix}{suffix}.pkl"],
                'actions': [(sheets_generation,
                             [transferred_path, sheets_path, suffix],
                             dict(succ_rolls_prefix=successful_rolls_prefix)
                             )],
                'verbosity': 2,
                # 'uptodate': [False]
            }


# To use for debugging
if __name__ == '__main__':
    g = globals()
    # run_tasks(ModuleTaskLoader(g), {'train:4-br': 1})
    run(g)
