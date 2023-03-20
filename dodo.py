import os.path
from copy import copy

from doit.api import run
from keras.models import load_model

from data_analysis.assemble_data import calculate_long_df, calculate_closest_styles, get_df_bigram_matrices
from data_analysis.dataset_plots import plot_styles_heatmaps_and_get_histograms, plot_distances_distribution, \
    plot_closeness, \
    plot_closest_ot_style, plot_styles_bigrams_entropy, heatmap_style_differences, plot_heatmap_differences, \
    plot_accuracy, plot_accuracy_distribution, plot_styles_confusion_matrix
from data_analysis.statistics import stratified_split, closest_ot_style, styles_bigrams_entropy, styles_ot_table
from evaluation.app.html_maker import make_html
from evaluation.evaluation import evaluate_model, evaluate_musicality
from evaluation.metrics.intervals import get_intervals_distribution
from evaluation.metrics.metrics import obtain_metrics
from evaluation.metrics.rhythmic_bigrams import get_rhythmic_distribution
from model.colab_tension_vae.params import init
from model.embeddings.characteristics import obtain_characteristics
from model.embeddings.embeddings import get_reconstruction, obtain_embeddings
from model.embeddings.style import Style
from model.embeddings.transfer import transfer_style_to
from model.train import train_model
from preprocessing import preprocess_data
from utils.audio_management import generate_audios
from utils.files_utils import *
from utils.plots_utils import calculate_TSNEs, plot_tsne, plot_tsnes_comparison, plot_embeddings, \
    plot_characteristics_distributions
from utils.utils import show_sheets
from utils.sampling_utils import sample_uniformly, balanced_sampling

subdatasets = ["Bach", "Mozart", "Frescobaldi", "ragtime"]
small_subdatasets = ["small_Bach", "small_ragtime"]
styles_dict = {'b': "Bach", 'm': "Mozart", 'f': "Frescobaldi", 'r': "ragtime"}

bars = [4]  # [4, 8]
# old_models = ['brmf_4b', 'brmf_8b']
old_models = ["brmf_4b", "brmf_4b_beta"]
models = old_models + [f"{b}-{x}{y}" for b in bars for x in 'brmf' for y in 'brmf' if x < y] + ["4-small_br"]


epochs = [200, 500, 1000]
checkpoints = [50, 100]
cross_val = 5


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

def prepare_data(df_path, eval_dir, b, cv):
    init(b)
    df = load_pickle(df_path)

    dfs_80, dfs_test = stratified_split(df, n_splits=cross_val)

    styles = set(df["Style"])
    if cv:
        for i, (df_80, df_test) in enumerate(zip(dfs_80, dfs_test)):
            styles_train = {name: Style(name, None, df_80) for name in styles}

            rolls_long_df_test = calculate_long_df(df, calculate_closest_styles(df_test, styles_train), styles_train)
            rolls_long_df_test.drop(columns=["roll"], inplace=True)

            save_pickle(rolls_long_df_test, f'{eval_dir}/rolls_long_df_test_{i}')
            save_pickle(df_80.index, f'{eval_dir}/df_80_indexes_{i}.pkl')
            rolls_long_df_test.to_csv(f'{eval_dir}/rolls_long_df_test_{i}.csv')

            # Musicality
            print("Calculating musicality distributions")
            melodic_distribution = get_intervals_distribution(df_80)
            rhythmic_distribution = get_rhythmic_distribution(df_80)

            save_pickle(melodic_distribution, f'{eval_dir}/melodic_distribution_{i}')
            save_pickle(rhythmic_distribution, f'{eval_dir}/rhythmic_distribution_{i}')
    else:
        df_to_analyze = get_df_bigram_matrices(df)
        save_pickle(df_to_analyze, f'{eval_dir}/df_to_analyze')

        print("Calculating musicality distributions")
        df_balanced = balanced_sampling(df)
        melodic_distribution = get_intervals_distribution(df_balanced)
        rhythmic_distribution = get_rhythmic_distribution(df_balanced)

        save_pickle(melodic_distribution, f'{eval_dir}/melodic_distribution')
        save_pickle(rhythmic_distribution, f'{eval_dir}/rhythmic_distribution')

def task_assemble_data_to_analyze():
    """Prepare the data for analysis"""
    for b in bars:
        for model in models:
            eval_dir = f"{data_path}{model}/Evaluation"
            yield {
                'name': f"{model}",
                'file_dep': [preprocessed_data(b)],
                'actions': [(prepare_data, [preprocessed_data(b), eval_dir, b, False])],
                'targets': [eval_dir + '/df_to_analyze.pkl',
                            eval_dir + '/melodic_distribution.pkl',
                            eval_dir + '/rhythmic_distribution.pkl'],
                # 'uptodate': [False]
            }

            eval_dir = f"{data_path}{model}/Evaluation/cross_val"
            yield {
                'name': f"{model}-cv",
                'file_dep': [preprocessed_data(b)],
                'actions': [(prepare_data, [preprocessed_data(b), eval_dir, b, True])],
                'targets': [eval_dir + '/df_80_indexes_0.pkl',
                            eval_dir + '/rolls_long_df_test_0.csv',
                            eval_dir + '/rolls_long_df_test_0.pkl',
                            eval_dir + '/melodic_distribution_0.pkl',
                            eval_dir + '/rhythmic_distribution_0.pkl'],
            }

def data_analysis(df_path, df_80_indexes_path, dfs_test_path, eval_dir, b, analysis, cv):
    init(b)
    df = load_pickle(df_path)
    styles = set(df["Style"])
    eval_dir_cv = eval_dir + '/cross_val'

    if cv:
        for i in range(cross_val):
            df_test = load_pickle(f'{dfs_test_path}{i}')
            df_80 = df.loc[load_pickle(f'{df_80_indexes_path}{i}')]

            if analysis == 'style_closeness':
                for orig in styles:
                    plot_closeness(df_test[df_test["Style"] == orig], orig, str(i), eval_dir_cv + "/styles")
                plot_accuracy(df_test, f'{eval_dir_cv}/{i}')
                plot_accuracy_distribution(dfs_test_path, eval_dir_cv)

            elif analysis in ['distances_distribution', 'style_differences', 'style_histograms']:
                histograms_80 = plot_styles_heatmaps_and_get_histograms(df_80, f'{eval_dir_cv}/{i}/80-percent')

                if analysis == 'style_differences':
                    diff_table_80 = styles_ot_table(df_80, histograms_80)
                    heatmap_style_differences(diff_table_80, f'{eval_dir_cv}/{i}/80-percent')
                    plot_heatmap_differences(df_80, histograms_80, f'{eval_dir_cv}/{i}/80-percent')

                elif analysis == 'distances_distribution':
                    rolls_diff_df = closest_ot_style(df_test, histograms_80)

                    plot_distances_distribution(rolls_diff_df, f'{eval_dir_cv}/{i}', by_style=False)
                    # plot_distances_distribution(rolls_diff_df, f'{eval_dir}/{i}', single_plot=True)

                    plot_closest_ot_style(rolls_diff_df, f'{eval_dir_cv}/{i}')
            
            elif analysis == 'musicality':
                melodic_distribution = load_pickle(f'{eval_dir_cv}/melodic_distribution_{i}')
                rhythmic_distribution = load_pickle(f'{eval_dir_cv}/rhythmic_distribution_{i}')

                evaluate_musicality(df_80, df_test, melodic_distribution, rhythmic_distribution, f'{eval_dir_cv}/{i}',
                                    '')

    else:
        if analysis == 'style_closeness':
            for orig in styles:
                plot_closeness(df[df["Style"] == orig], orig, 'nothing', eval_dir + "/styles")
            plot_accuracy(df, eval_dir)

        elif analysis in ['distances_distribution', 'style_differences', 'style_histograms', 'confusion_matrix']:
            histograms = plot_styles_heatmaps_and_get_histograms(df, eval_dir)
            save_pickle(histograms, eval_dir + '/style_histograms', verbose=True)

            if analysis == 'style_differences':
                diff_table = styles_ot_table(df, histograms)
                heatmap_style_differences(diff_table, eval_dir)
                plot_heatmap_differences(df, histograms, eval_dir)

            elif analysis == 'distances_distribution':
                rolls_diff_df = closest_ot_style(df, histograms)

                plot_distances_distribution(rolls_diff_df, eval_dir, by_style=False)
                # plot_distances_distribution(rolls_diff_df, f'{eval_dir}/{i}', single_plot=True)

                plot_closest_ot_style(rolls_diff_df, eval_dir)

            elif analysis == 'confusion_matrix':
                rolls_diff_df = closest_ot_style(df, histograms)
                plot_styles_confusion_matrix(rolls_diff_df, styles, eval_dir)

        elif analysis == 'entropies':
            entropies = styles_bigrams_entropy(df)
            plot_styles_bigrams_entropy(entropies, eval_dir)



def task_analyze_data():
    """Get different kind of analysis of the dataset ('style_closeness', 'distances_distribution', 'musicality', 'entropies', 'style_histograms', 'confusion_matrix' and 'style_differences')"""
    for b in bars:
        for analysis in ['style_closeness', 'distances_distribution', 'entropies', 'style_differences', 'musicality', 'style_histograms', 'confusion_matrix']:
            for cv in [True, False]:
                for model in old_models:
                    eval_dir = f"{data_path}{model}/Evaluation"
                    df_80_indexes_path = eval_dir + '/cross_val/df_80_indexes_'
                    df_test_path = eval_dir + '/cross_val/rolls_long_df_test_'
                    yield {
                        'name': f"{model}-{analysis}{'-cv' if cv else ''}",
                        'file_dep': [eval_dir + '/df_to_analyze.pkl',
                                     eval_dir + f"/cross_val/df_80_indexes{'_0' if cv else ''}.pkl",
                                     # eval_dir + f"/cross_val/rolls_long_df_test{'_0' if cv else ''}.pkl",
                                     eval_dir + f"/cross_val/melodic_distribution{'_0' if cv else ''}.pkl",
                                     eval_dir + f"/cross_val/rhythmic_distribution{'_0' if cv else ''}.pkl"
                                     ],
                        'actions': [(data_analysis, [f'{eval_dir}/df_to_analyze',
                                                     df_80_indexes_path,
                                                     df_test_path,
                                                     eval_dir,
                                                     b,
                                                     analysis,
                                                     cv])],
                        'targets': [eval_dir + '/style_histograms.pkl'
                                    if analysis == 'style_differences' and not cv
                                    else f'{eval_dir}/{analysis}{cv}'],
                        # 'uptodate': [False]
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
        b = model_name[5] if model_name in old_models else model_name[0]
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
        b = model_name[5] if model_name in old_models else model_name[0]
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
    model = load_model(os.path.abspath(vae_path))
    plots_dir = os.path.join(model_path, "plots")
    df = load_pickle(df_path)

    df_emb, styles_char = obtain_characteristics(df, model)

    plot_embeddings(df_emb, "Embedding", {n: s.embedding for n, s in styles_char.items()}, plots_dir,
                    include_songs=True)
    # plot_characteristics_distributions(styles_char, plots_dir, "Distributions_characteristics")

    save_pickle(styles_char, characteristics_path)
    save_pickle(df_emb, emb_path)


def task_embeddings():
    """Calculate the embeddings for each author/style and song"""
    for model_name in models:
        b = model_name[5] if model_name in old_models else model_name[0]
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
            # 'uptodate': [os.path.isfile(characteristics_path) and os.path.isfile(emb_path)]
            'uptodate': [False]
        }


def do_transfer(df_emb, model_path, characteristics, transferred_path, s1, s2, b=4):
    init(b)
    df_emb = load_pickle(df_emb)
    model = load_model(model_path)
    characteristics = load_pickle(characteristics)
    model_name = os.path.basename(model_path)

    df_transferred = transfer_style_to(df_emb, model, model_name, characteristics, original_style=s1, target_style=s2)

    save_pickle(df_transferred, transferred_path)


def task_transfer_style():
    """Do the transference of style from a roll to another style"""
    for model_name in models:
        b = model_name[5] if model_name in old_models else model_name[0]
        model_path, vae_dir, vae_path = get_model_paths(model_name)
        characteristics_path = get_characteristics_path(model_name)
        emb_path = get_emb_path(model_name)

        for s1, s2 in styles_names(model_name):
            transferred_path = get_transferred_path(s1, s2, model_name)
            yield {
                'name': f"{model_name}_{s1}_to_{s2}",
                'file_dep': [emb_path, vae_path, characteristics_path],
                'actions': [(do_transfer,
                             [emb_path, vae_dir, characteristics_path, transferred_path, s1, s2, b]
                             )],
                'targets': [transferred_path],
                'verbosity': 2,
                # 'uptodate': [False]
            }


def calculate_metrics(trans_path, char_path, metrics_dir, s1, s2, b=4):
    init(b)
    df_transferred = load_pickle(trans_path)
    styles = load_pickle(char_path)

    metrics1 = obtain_metrics(df_transferred, s1, s2, 'plagiarism', 'intervals', 'rhythmic_bigrams')
    save_pickle(metrics1, f"{metrics_dir}/metrics_{s1}_to_{s2}")


def task_metrics():
    """Calculate different metrics for a produced dataset"""
    for model_name in models:
        b = model_name[5] if model_name in old_models else model_name[0]

        for s1, s2 in styles_names(model_name):
            transferred_path = get_transferred_path(s1, s2, model_name)
            characteristics_path = get_characteristics_path(model_name)
            metrics_path = get_metrics_dir(transferred_path)
            yield {
                'name': f"{model_name}_{s1}_to_{s2}",
                'file_dep': [transferred_path, characteristics_path],
                'actions': [(calculate_metrics, [transferred_path, characteristics_path, metrics_path, s1, s2, b])],
                'targets': [f"{metrics_path}/metrics_{s1}_to_{s2}.pkl"],
                'verbosity': 2,
                # 'uptodate': [False]
            }


def do_evaluation(trans_path, styles_path, eval_dir, s1, s2, b=4):
    init(b)
    metrics_dir = get_metrics_dir(trans_path)

    df_transferred = load_pickle(trans_path)
    styles = load_pickle(styles_path)

    metrics = load_pickle(f"{metrics_dir}/metrics_{s1}_to_{s2}")
    melodic_musicality_distribution = load_pickle(eval_dir + '/melodic_distribution.pkl')
    rhythmic_musicality_distribution = load_pickle(eval_dir + '/rhythmic_distribution.pkl')

    successful_rolls, table = evaluate_model(df_transferred, metrics, styles, melodic_musicality_distribution,
                                             rhythmic_musicality_distribution, eval_path=eval_dir)
    save_pickle(successful_rolls, f"{eval_dir}/successful_rolls-{s1}_to_{s2}")
    save_pickle(table, f"{eval_dir}/results-{s1}_to_{s2}")
    for t in table.values():
        print(t)


def task_evaluation():
    """Evaluate the model considering the calculated metrics"""
    for model_name in models:
        b = model_name[5] if model_name in old_models else model_name[0]
        for s1, s2 in styles_names(model_name):
            transferred_path = get_transferred_path(s1, s2, model_name)
            styles_path = get_characteristics_path(model_name)
            metrics_dir = get_metrics_dir(transferred_path)
            eval_dir = get_eval_dir(transferred_path)
            yield {
                'name': f"{model_name}_{s1}_to_{s2}",
                'file_dep': [transferred_path, styles_path,
                             f"{metrics_dir}/metrics_{s1}_to_{s2}.pkl", f"{metrics_dir}/metrics_{s2}_to_{s1}.pkl",
                             eval_dir + '/melodic_distribution.pkl',
                             eval_dir + '/rhythmic_distribution.pkl'
                             ],
                'actions': [(do_evaluation, [transferred_path, styles_path, eval_dir, s1, s2, b])],
                'targets': [f"{eval_dir}/successful_rolls-{s1}_to_{s2}.pkl"],
                'verbosity': 2,
                'uptodate': [False]
            }


def audio_generation(transferred_path, audios_path, succ_rolls_prefix=None,
                     suffix=None, orig=None, dest=None, b=4):
    init(b)
    if succ_rolls_prefix is None:
        df_transferred = load_pickle(transferred_path)
        generate_audios(df_transferred, audios_path, suffix=suffix, verbose=1)
        make_html(df_transferred, orig=orig, target=dest, app_dir=audios_path)
    else:
        successful_dfs = load_pickle(f"{succ_rolls_prefix}{suffix}")
        df_html = pd.DataFrame()
        for k, df in successful_dfs.items():
            df = sample_uniformly(df[df["Style"] == orig], f"{k} rank", n=3)
            original_files, new_files = generate_audios(df, audios_path, f"{k}-{suffix}", 1)
            df["Original audio files"] = original_files
            df["New audio files"] = new_files
            df["Selection criteria"] = len(new_files) * [k]
            df_html = pd.concat([df_html, df])

        df = load_pickle(transferred_path)
        df = df[df["Style"] == orig].sample(n=4, random_state=42)

        original_files, new_files = generate_audios(df, audios_path, f"random-{suffix}", 1)
        df["Original audio files"] = original_files
        df["New audio files"] = new_files
        df["Selection criteria"] = len(new_files) * ["random"]
        df_html = pd.concat([df_html, df])

        make_html(df_html, orig=orig, target=dest, app_dir=os.path.dirname(os.path.dirname(audios_path)) + '/app')
        df_html[["Title", "Style", "target", "Selection criteria", "Original audio files", "New audio files"]].to_csv(
            f'{os.path.dirname(os.path.dirname(audios_path))}/app/Sampled_rolls-{orig}_to_{dest}.csv')

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
                'file_dep': [f"{successful_rolls_prefix}{suffix}.pkl"],
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
