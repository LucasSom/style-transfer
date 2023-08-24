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
from evaluation.html_maker import make_html, make_index
from evaluation.evaluation import evaluate_model, evaluate_musicality
from evaluation.metrics.intervals import get_intervals_distribution
from evaluation.metrics.metrics import obtain_metrics
from evaluation.metrics.rhythmic_bigrams import get_rhythmic_distribution
from evaluation.overall_evaluation import overall_evaluation
from model.colab_tension_vae.params import init
from model.embeddings.characteristics import obtain_characteristics, interpolate_centroids
from model.embeddings.embeddings import get_reconstruction
from model.embeddings.style import Style
from model.embeddings.transfer import transfer_style_to
from model.train import train_model
from preprocessing.preprocessing import preprocess_data, oversample
from utils.audio_management import generate_audios
from utils.files_utils import *
from utils.files_utils import preprocessed_data_path
from utils.plots_utils import plot_embeddings
from utils.sampling_utils import sample_uniformly, balanced_sampling
from utils.utils import generate_sheets

DOIT_CONFIG = {'verbosity': 2}

subdatasets = ["Bach", "Mozart", "Frescobaldi", "ragtime"]
subdataset_lmd = "sub_lmd"
subdatasets_lmd = ["sub_lmd/classical", 'sub_lmd/pop', 'sub_lmd/rock', 'sub_lmd/folk', 'sub_lmd/cpr1']
small_subdatasets = ["small_Bach", "small_ragtime"]
styles_dict = {'b': "Bach", 'm': "Mozart", 'f': "Frescobaldi", 'r': "ragtime", "C": "sub_lmd/classical",
               "P": "sub_lmd/pop", "F": "sub_lmd/folk", "R": "sub_lmd/rock", "A": "sub_lmd/cpr0", "a": "sub_lmd/cpr1"}

# z_dims = [20, 96, 192, 512]
z_dims = [96]
bars = [4]  # [4, 8]
# old_models = ['brmf_4b', 'brmf_8b']
old_models = [f"brmf_4b-{z}" for z in z_dims] + [f"brmf_4b_beta-{z}" for z in z_dims]
pre_models = [f"4-CPFRAa-{z}" for z in z_dims]
mixture_models = [f"4-Lakh_Kern-{z}" for z in z_dims]
ensamble_models = [f"{b}-{x}{y}-{z}" for z in z_dims for b in bars for x in 'brmf' for y in 'brmf' if x < y]
models = old_models + ensamble_models + ["4-small_br-96"] + pre_models + mixture_models

alphas = [0.1, 0.25, 0.5, 0.75, 1]
mutations_add = [f'Mutation_add_{a}' for a in alphas]
mutations_add_sub = [f'Mutation_add_sub_{a}' for a in alphas]
mutations = mutations_add + mutations_add_sub

epochs = [200, 500, 1000]
checkpoints = [50, 100]
cross_val = 5


def styles_names(model_name):
    if model_name in ensamble_models:
        m1 = styles_dict[model_name.split('-')[1][0]]
        m2 = styles_dict[model_name.split('-')[1][1]]
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


def preprocess(b, folders, save_midis, sparse, targets):
    init(b)
    songs = {}
    for folder in folders:
        songs[folder] = [f"{folder}/{song}"
                         for song in os.listdir(f"{datasets_path}/{folder}")]
    df = preprocess_data(songs, save_midis, sparse=sparse)
    save_pickle(df, targets[0], verbose=True)


def task_preprocess():
    """Preprocess dataset, considering the subdatasets referenced in the list at the top of this file"""
    # files = [f for sd in subdatasets for f in list(pathlib.Path(sd).glob('*.txt'))]
    for b in bars:
        yield {
            # 'file_dep': files,
            'name': f"{b}bars",
            'actions': [(preprocess, [b], {'folders': subdatasets, 'save_midis': True, 'sparse': False})],
            'targets': [preprocessed_data_path(b, False)],
            'uptodate': [os.path.isfile(preprocessed_data_path(b, False))]
        }
        # for i in range(32):
        #     yield {
        #         'name': f"{b}bars_lmd-{i}",
        #         'actions': [(preprocess, [b, [f'{subdataset_lmd}/{i}'], False, True])],
        #         'targets': [preprocessed_data_path(b, i + 1)],
        #         'uptodate': [os.path.isfile(preprocessed_data_path(b, False))]
        #     }
        yield {
            'name': f"{b}bars_sublmd",
            'actions': [(preprocess, [b, subdatasets_lmd, False, True])],
            'targets': [preprocessed_data_path(b, True)],
            'uptodate': [os.path.isfile(preprocessed_data_path(b, False))]
        }
    yield {
        # 'file_dep': files,
        'name': f"small_4bars",
        'actions': [(preprocess, [4], {'folders': small_subdatasets, 'save_midis': False, 'sparse': True})],
        'targets': [preprocessed_data_path(4, False, True)],
        # 'uptodate': [os.path.isfile(preprocessed_data_path(4, False, True))]
        'uptodate': [False]
    }


def split(b, df_path, train_path, test_path, val_path):
    init(b)

    df = load_pickle(df_path)
    dfs_train, dfs_test_val = stratified_split(df)
    df_test_val = dfs_test_val[0]
    dfs_test, dfs_val = stratified_split(df_test_val, test_size=0.5)

    save_pickle(dfs_train[0], train_path)
    save_pickle(dfs_test[0], test_path)
    save_pickle(dfs_val[0], val_path)


def task_split_dataset():
    """Split the dataset into train, test and validation"""
    for model in models:
        train_path = f"{preprocessed_data_dir}{model}train.pkl"
        test_path = f"{preprocessed_data_dir}{model}test.pkl"
        val_path = f"{preprocessed_data_dir}{model}val.pkl"
        df_path = preprocessed_data_path(4, (model in pre_models), False)
        yield {
            'file_dep': [df_path],
            'name': model,
            'actions': [(split, [4, df_path, train_path, test_path, val_path])],
            'targets': [train_path, test_path, val_path],
            # 'uptodate': [False],
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
        styles_train = {name: Style(name, None, dfs_80[0]) for name in styles}
        df_to_analyze = calculate_closest_styles(df_to_analyze, styles_train, only_ot=True)
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
        eval_dir = f"{data_path}/data_analysis"
        yield {
            'name': "not_cv",
            'file_dep': [preprocessed_data_path(b, False)],
            'actions': [(prepare_data, [preprocessed_data_path(b, False), eval_dir, b, False])],
            'targets': [eval_dir + '/df_to_analyze.pkl',
                        eval_dir + '/melodic_distribution.pkl',
                        eval_dir + '/rhythmic_distribution.pkl'],
            # 'uptodate': [False]
        }

        eval_dir = f"{data_path}data_analysis/cross_val"
        yield {
            'name': f"cv",
            'file_dep': [preprocessed_data_path(b, False)],
            'actions': [(prepare_data, [preprocessed_data_path(b, False), eval_dir, b, True])],
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
                    plot_closeness(df_test[df_test["Style"] == orig], orig, str(i), "dataset", eval_dir_cv + "/styles")
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
                plot_closeness(df[df["Style"] == orig], orig, 'nothing', "dataset", eval_dir + "/styles",
                               only_joined_ot=True)
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

                # plot_distances_distribution(rolls_diff_df, eval_dir, by_style=False)
                # # plot_distances_distribution(rolls_diff_df, f'{eval_dir}/{i}', single_plot=True)

                plot_closest_ot_style(rolls_diff_df, eval_dir)
                save_pickle(rolls_diff_df, eval_dir + '/rolls_diff_df')

            elif analysis == 'confusion_matrix':
                try:
                    rolls_diff_df = load_pickle(eval_dir + '/rolls_diff_df')
                except:
                    rolls_diff_df = closest_ot_style(df, histograms)
                plot_styles_confusion_matrix(rolls_diff_df, styles, eval_dir)

        elif analysis == 'entropies':
            entropies = styles_bigrams_entropy(df)
            plot_styles_bigrams_entropy(entropies, eval_dir, english=False)

        elif analysis == 'style_confusion_matrix':
            pass

        elif analysis == 'musicality':
            df_test = load_pickle(f'{dfs_test_path}0')
            df_80 = df.loc[load_pickle(f'{df_80_indexes_path}0')]

            melodic_distribution = load_pickle(f'{eval_dir_cv}/melodic_distribution_0')
            rhythmic_distribution = load_pickle(f'{eval_dir_cv}/rhythmic_distribution_0')

            evaluate_musicality(df_80, df_test, melodic_distribution, rhythmic_distribution, f'{eval_dir}', '')


def task_analyze_data():
    """Get different kind of analysis of the dataset ('style_closeness', 'distances_distribution', 'musicality',
    'entropies', 'style_histograms', 'confusion_matrix', 'style_differences' and 'style_confusion_matrix')"""
    for b in bars:
        for analysis in ['style_closeness', 'distances_distribution', 'entropies', 'style_differences', 'musicality',
                         'style_histograms', 'confusion_matrix', 'style_confusion_matrix']:
            for cv in [True, False]:
                eval_dir = f"{data_path}data_analysis"
                df_80_indexes_path = eval_dir + '/cross_val/df_80_indexes_'
                df_test_path = eval_dir + '/cross_val/rolls_long_df_test_'
                yield {
                    'name': f"{analysis}{'-cv' if cv else ''}",
                    'file_dep': [eval_dir + '/df_to_analyze.pkl',
                                 eval_dir + f"/{'cross_val/df_80_indexes_0' if cv else 'df_to_analyze'}.pkl",
                                 eval_dir + f"/{'cross_val/' if cv else ''}rolls_long_df_test{'_0' if cv else ''}.pkl",
                                 eval_dir + f"/{'cross_val/' if cv else ''}melodic_distribution{'_0' if cv else ''}.pkl",
                                 eval_dir + f"/{'cross_val/' if cv else ''}rhythmic_distribution{'_0' if cv else ''}.pkl"
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


def do_oversampling(path_in, path_out, b):
    init(b)
    df = load_pickle(path_in)
    df = oversample(df)
    save_pickle(df, path_out)


def task_oversample():
    """Balances the minority classes"""
    for model_name in models:
        b = model_name[5] if model_name in old_models else model_name[0]
        train_path = f"{preprocessed_data_dir}{model_name}train.pkl"
        oversample_data_path = oversample_path(model_name)
        yield {
            'name': f"{model_name}",
            'file_dep': [train_path],
            'actions': [(do_oversampling, [train_path, oversample_data_path, b])],
            'targets': [oversample_data_path],
            # 'uptodate': [True]
        }


def train(train_path, test_path, model_name, b, z, debug=False):
    init(b, z)
    if_train = input("Do you want to train the model [Y/n]? ") if not debug else 'Y'
    if if_train in ['Y', 'y', 'S', 's']:
        if "small" in model_name:
            styles = ["small_Bach", "small_ragtime"]
        elif model_name in old_models:
            styles = [styles_dict[a] for a in model_name[0:4]]
        elif model_name in pre_models:
            styles = [styles_dict[a] for a in model_name[2:8]]
        elif model_name in mixture_models:
            model_name_aux = pre_models[0]
            styles = [styles_dict[a] for a in model_name_aux[2:8]]
        else:
            styles = [styles_dict[a] for a in model_name[2:4]]

        df = load_pickle(train_path)
        df = df[df['Style'].isin(styles)]
        test_data = load_pickle(test_path)
        test_data = test_data[test_data['Style'].isin(styles)]

        train_model(df, test_data, model_name, debug=debug)
    elif if_train in ['N', 'n']:
        print("Skipping training")
    else:
        print("Try again")


def task_train():
    """Trains the model"""
    for model_name in models:
        b = model_name[5] if model_name in old_models else model_name[0]
        z = int(model_name.split("-")[-1])

        oversample_data_path = oversample_path(model_name)
        test_path = f"{preprocessed_data_dir}{model_name}train.pkl"

        vae_path = get_model_paths(model_name)[2]
        if "small" in model_name:
            oversample_data_path = f"{preprocessed_data_dir}4-small_br.pkl"
            test_path = oversample_data_path
        if model_name in pre_models:
            oversample_data_path = f"{preprocessed_data_dir}{model_name}train.pkl"
        if model_name in mixture_models:
            model_name_aux = f"{b}-CPFRAa-{z}"
            oversample_data_path = f"{preprocessed_data_dir}{model_name_aux}train"
            test_path = f"{preprocessed_data_dir}{model_name_aux}train.pkl"
        yield {
            'name': f"{model_name}",
            'file_dep': [oversample_data_path, test_path],
            'actions': [(train, [oversample_data_path, test_path, model_name, b, z])],
            'targets': [vae_path],
            # 'uptodate': [True]
        }


def analyze_training(train_path, model_name, vae_dir, b, z, targets):
    init(b, z)
    # model_path = get_model_paths(model_name)[0]
    model = load_model(vae_dir)
    model_name = os.path.basename(model_name)
    # plots_path = os.path.join(data_path, model_path, "plots")
    audios_path = get_audios_path(model_name)
    # logs_path = get_logs_path(model_name)
    train_df = load_pickle(train_path)
    # val_df = load_pickle(val_path)

    # plot_accuracies(val_df, model, logs_path)

    df_emb, styles = obtain_characteristics(train_df, model)
    if model_name in mixture_models:
        styles = {name.split('/')[-1]: s for name, s in styles.items()}
    df_interpolation = interpolate_centroids(styles.values(), model, audios_path + 'interpolation/')
    save_pickle(df_interpolation, targets[0] + '-interpolation')

    # tsne_emb = calculate_TSNEs(df_emb, column_discriminator="Style")[0]

    # plot_tsnes_comparison(df_emb, tsne_emb, plots_path)
    # plot_tsne(df_emb, tsne_emb, plots_path)


def task_test():
    """Shows a t-SNE plot of the songs in the latent space."""
    for model_name in models:
        b = model_name[5] if model_name in old_models else model_name[0]
        z = int(model_name.split("-")[-1])

        if model_name in mixture_models:
            model_name_aux = f"{b}-CPFRAa-{z}"
            _, vae_dir, vae_path = get_model_paths(model_name_aux)
            train_path = f"{preprocessed_data_dir}{model_name_aux}train.pkl"
        else:
            _, vae_dir, vae_path = get_model_paths(model_name)
            train_path = f"{preprocessed_data_dir}{model_name}train.pkl"

        yield {
            'name': model_name,
            'file_dep': [train_path, vae_path],
            'actions': [(analyze_training, [train_path, model_name, vae_dir, b, z])],
            # 'uptodate': [False]
        }


def do_embeddings(df_path, model_path, vae_path, characteristics_path, emb_path, b, z):
    init(b, z)
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
        z = int(model_name.split("-")[-1])

        if model_name in mixture_models:
            model_name_model = f"{b}-CPFRAa-{z}"
            model_name_data = f"brmf_{b}b_beta-{z}"
            model_path, vae_dir, vae_path = get_model_paths(model_name_model)
            train_path = f"{preprocessed_data_dir}{model_name_data}train.pkl"
        else:
            model_path, vae_dir, vae_path = get_model_paths(model_name)
            train_path = f"{preprocessed_data_dir}{model_name}train.pkl"

        characteristics_path = get_characteristics_path(model_name)
        emb_path = get_emb_path(model_name)

        yield {
            'name': model_name,
            'file_dep': [train_path, vae_path],
            'actions': [(do_embeddings,
                         [train_path,
                          model_path,
                          vae_dir,
                          characteristics_path,
                          emb_path,
                          b, z]
                         )],
            'targets': [characteristics_path, emb_path],
            'uptodate': [os.path.isfile(characteristics_path) and os.path.isfile(emb_path)]
            # 'uptodate': [False]
        }


def do_reconstructions(emb_path, model_name, vae_dir, b, z, targets):
    init(b, z)
    model = load_model(vae_dir)
    df_emb = load_pickle(emb_path)

    df_reconstructed = get_reconstruction(df_emb, model, model_name)
    save_pickle(df_reconstructed, targets[0])


def task_reconstruct():
    """Generates the reconstruction of the original rolls after being passed through the model."""
    for model_name in models:
        b = model_name[5] if model_name in old_models else model_name[0]
        z = int(model_name.split("-")[-1])

        if model_name in mixture_models:
            model_name_aux = f"{b}-CPFRAa-{z}"
            _, vae_dir, vae_path = get_model_paths(model_name_aux)
        else:
            _, vae_dir, vae_path = get_model_paths(model_name)

        emb_path = get_emb_path(model_name)

        yield {
            'name': model_name,
            'file_dep': [emb_path, vae_path],
            'actions': [(do_reconstructions, [emb_path, model_name, vae_dir, b, z])],
            'targets': [get_reconstruction_path(model_name)],
            # 'uptodate': [False]
        }


def do_transfer(rec_path, model_path, characteristics, transferred_path, s1, s2, b=4, z=96):
    init(b, z)
    df_rec = load_pickle(rec_path)
    model = load_model(model_path)
    characteristics = load_pickle(characteristics)
    model_name = os.path.basename(model_path)

    df_transferred = transfer_style_to(df_rec, model, model_name, characteristics, original_style=s1, target_style=s2,
                                       alphas=alphas, mutations=mutations, sparse=False)

    save_pickle(df_transferred, transferred_path)


def task_transfer_style():
    """Do the transference of style from a roll to another style"""
    for model_name in models:
        b = model_name[5] if model_name in old_models else model_name[0]
        z = int(model_name.split("-")[-1])

        if model_name in mixture_models:
            model_name_model = f"{b}-CPFRAa-{z}"
            _, vae_dir, vae_path = get_model_paths(model_name_model)
        else:
            _, vae_dir, vae_path = get_model_paths(model_name)

        characteristics_path = get_characteristics_path(model_name)
        rec_path = get_reconstruction_path(model_name)

        for s1, s2 in styles_names(model_name):
            transferred_path = get_transferred_path(s1, s2, model_name)
            yield {
                'name': f"{model_name}_{s1}_to_{s2}",
                'file_dep': [rec_path, vae_path, characteristics_path],
                'actions': [(do_transfer,
                             [rec_path, vae_dir, characteristics_path, transferred_path, s1, s2, b, z]
                             )],
                'targets': [transferred_path],
                'verbosity': 2,
                # 'uptodate': [False]
            }


def calculate_metrics(trans_path, metrics_dir, s1, s2, mutation, b=4, z=96):
    init(b, z)
    df_transferred = load_pickle(trans_path)

    metrics = obtain_metrics(df_transferred, s1, s2, mutation, 'plagiarism', 'intervals', 'rhythmic_bigrams')
    save_pickle(metrics, f"{metrics_dir}/metrics_{mutation}_{s1}_to_{s2}")


def task_metrics():
    """Calculate different metrics for a produced dataset"""
    for model_name in models:
        b = model_name[5] if model_name in old_models else model_name[0]
        z = int(model_name.split("-")[-1])

        for s1, s2 in styles_names(model_name):
            transferred_path = get_transferred_path(s1, s2, model_name)
            metrics_path = get_metrics_dir(model_name)
            for mutation in mutations:
                yield {
                    'name': f"{model_name}_{s1}_to_{s2}-{mutation}",
                    'file_dep': [transferred_path],
                    'actions': [(calculate_metrics, [transferred_path, metrics_path, s1, s2, mutation, b, z])],
                    'targets': [f"{metrics_path}/metrics_{mutation}_{s1}_to_{s2}.pkl"],
                    'verbosity': 2,
                    # 'uptodate': [False]
                }


def do_evaluation(trans_path, styles_path, metrics_dir, eval_dir, s1, s2, mutation, b=4, z=96):
    init(b, z)

    df_transferred = load_pickle(trans_path)
    styles = load_pickle(styles_path)

    metrics = load_pickle(f"{metrics_dir}/metrics_{mutation}_{s1}_to_{s2}")
    melodic_musicality_distribution = load_pickle(data_path + 'data_analysis/melodic_distribution.pkl')
    rhythmic_musicality_distribution = load_pickle(data_path + 'data_analysis/rhythmic_distribution.pkl')

    successful_rolls, tables, overall_metrics = evaluate_model(df_transferred, metrics, styles,
                                                               melodic_musicality_distribution,
                                                               rhythmic_musicality_distribution, mutation,
                                                               eval_path=eval_dir, plot=False)
    save_pickle(successful_rolls, f"{eval_dir}/successful_rolls-{mutation}-{s1}_to_{s2}")
    save_pickle(overall_metrics, f"{eval_dir}/overall_metrics_dict-{mutation}-{s1}_to_{s2}")
    for t, v in tables.items():
        v.to_csv(f"{eval_dir}/{t}_results-{mutation}-{s1}_to_{s2}.csv")


def task_evaluation():
    """Evaluate the model considering the calculated metrics"""
    for model_name in models:
        b = model_name[5] if model_name in old_models else model_name[0]
        z = int(model_name.split("-")[-1])
        for s1, s2 in styles_names(model_name):
            transferred_path = get_transferred_path(s1, s2, model_name)
            styles_path = get_characteristics_path(model_name)
            metrics_dir = get_metrics_dir(model_name)
            eval_dir = get_eval_dir(model_name)
            for mutation in mutations:
                yield {
                    'name': f"{model_name}_{s1}_to_{s2}-{mutation}",
                    'file_dep': [transferred_path, styles_path,
                                 f"{metrics_dir}/metrics_{mutation}_{s1}_to_{s2}.pkl",
                                 f"{metrics_dir}/metrics_{mutation}_{s2}_to_{s1}.pkl",
                                 data_path + 'data_analysis/melodic_distribution.pkl',
                                 data_path + 'data_analysis/rhythmic_distribution.pkl'
                                 ],
                    'actions': [(do_evaluation,
                                 [transferred_path, styles_path, metrics_dir, eval_dir, s1, s2, mutation, b, z])],
                    'targets': [f"{eval_dir}/successful_rolls-{mutation}-{s1}_to_{s2}.pkl",
                                f"{eval_dir}/results-{mutation}-{s1}_to_{s2}.csv",
                                f"{eval_dir}/overall_metrics_dict-{mutation}-{s1}_to_{s2}.pkl"
                                ],
                    'verbosity': 2,
                    # 'uptodate': [False]
                }


def do_overall_evaluation(overall_metric_dirs, mutation, eval_dir, b=4, z=96):
    init(b, z)
    m = get_packed_metrics(overall_metric_dirs, mutation)
    overall_evaluation(m, eval_dir)


def task_overall_evaluation():
    """Calculate the final metrics after evaluate the model"""
    for z in z_dims:
        for b in bars:
            ensamble = [m for m in models if len(m) == 7 and m[0] == str(b) and m.split("-")[-1] == str(z)]
            overall_metric_dirs = [get_eval_dir(model_name) for model_name in ensamble]
            for mutation in mutations:
                eval_path = f"{data_path}/overall_evaluation/ensamble_{b}bars_{z}dim-{mutation}"
                yield {
                    'name': f"ensamble_{b}bars_{z}dim-{mutation}",
                    'file_dep': [f"{eval_dir}/overall_metrics_dict-{mutation}-{s1}_to_{s2}.pkl"
                                 for eval_dir in overall_metric_dirs for s1, s2 in styles_names("brmf_4b")
                                 ],
                    'actions': [(do_overall_evaluation, [overall_metric_dirs, mutation, eval_path, b, z])],
                    'targets': [],
                    'verbosity': 2,
                    'uptodate': [False]
                }

    for model_name in old_models + mixture_models:
        b = model_name[5] if model_name in old_models else model_name[0]
        z = int(model_name.split("-")[-1])
        eval_dir = get_eval_dir(model_name)
        for mutation in mutations:
            eval_path = f"{data_path}/overall_evaluation/{model_name}-{mutation}"
            yield {
                'name': f"{model_name}-{mutation}",
                'file_dep': [f"{eval_dir}/overall_metrics_dict-{mutation}-{s1}_to_{s2}.pkl"
                             for s1, s2 in styles_names(model_name)
                             ],
                'actions': [(do_overall_evaluation, [[eval_dir], mutation, eval_path, b, z])],
                'targets': [],
                'verbosity': 2,
                'uptodate': [False]
            }


def audio_generation(mutation, eval_dir, transferred_path, audios_path, succ_rolls_prefix=None, transformation=None,
                     b=4, z=96):
    init(b, z)
    orig = transformation.split('_')[0]
    successful_dfs = load_pickle(f"{succ_rolls_prefix}-{transformation}")
    df_audios = pd.DataFrame()
    df = load_pickle(transferred_path)

    for k, df_succ in successful_dfs.items():
        df_merged = df_succ.merge(df, how='inner')
        df_merged = sample_uniformly(df_merged[df_merged["Style"] == orig], f"{k} rank", n=5)
        original_files, reconstructed_files, new_files = generate_audios(df_merged, mutation, audios_path,
                                                                         f"{k}-{transformation}-{mutation}", 1)
        df_merged = df_merged[['Style', 'Title', 'roll_id']]
        df_merged["Original audio files"] = original_files
        df_merged["Reconstructed audios"] = reconstructed_files
        df_merged["New audio files"] = new_files
        df_merged["Selection criteria"] = len(new_files) * [k]
        df_audios = pd.concat([df_audios, df_merged])

    # Include random selection
    df = df[df["Style"] == orig].sample(n=5, random_state=42)
    original_files, reconstructed_files, new_files = generate_audios(df, mutation, audios_path,
                                                                     f"random-{transformation}-{mutation}", 1)
    sub_df = df[['Style', 'Title', 'roll_id']]
    sub_df["Original audio files"] = original_files
    sub_df["Reconstructed audios"] = reconstructed_files
    sub_df["New audio files"] = new_files
    sub_df["Selection criteria"] = len(new_files) * ["random"]
    df_audios = pd.concat([df_audios, sub_df])

    save_pickle(df_audios, f"{eval_dir}/df_audios-{transformation}-{mutation}")


def task_sample_audios():
    """Produce the midis generated by the style transfer"""
    for model_name in models:
        audios_path = get_audios_path(model_name)
        b = model_name[5] if model_name in old_models else model_name[0]
        z = int(model_name.split("-")[-1])

        for s1, s2 in styles_names(model_name):
            transferred_path = get_transferred_path(s1, s2, model_name)
            eval_dir = get_eval_dir(model_name)
            transformation = f'{s1}_to_{s2}'
            for mutation in mutations:
                successful_rolls_prefix = f"{eval_dir}/successful_rolls-{mutation}"
                yield {
                    'name': f"{model_name}-{transformation}-{mutation}",
                    'file_dep': [f"{successful_rolls_prefix}-{transformation}.pkl", transferred_path],
                    'actions': [(audio_generation,
                                 [mutation, eval_dir, transferred_path, audios_path, successful_rolls_prefix,
                                  transformation, b, z])],
                    'targets': [f"{eval_dir}/df_audios-{transformation}-{mutation}.pkl"],
                    'verbosity': 2,
                    # 'uptodate': [False]
                }


def sheets_generation(sheets_path, transferred_path, transference, mutation, df_audios_path, df_sheets_paths, b=4,
                      z=96):
    init(b, z)
    df_transferred = load_pickle(transferred_path)
    df_audios = load_pickle(df_audios_path)
    df = df_audios.merge(df_transferred, how='inner')

    dest = transference.split('_')[-1]
    original_sheets = generate_sheets(df, 'roll', sheets_path, suffix='')
    reconstructed_sheets = generate_sheets(df, 'Reconstruction', sheets_path, suffix='-rec')
    new_sheets = generate_sheets(df, f'{mutation}-NewRoll', sheets_path, suffix=f'-{dest}')

    df_audios["Original sheet"] = original_sheets
    df_audios["Reconstructed sheet"] = reconstructed_sheets
    df_audios["New sheet"] = new_sheets

    save_pickle(df_audios, df_sheets_paths)


def task_sample_sheets():
    """Produces the sheets generated by the style transfer"""
    for model_name in models:
        b = model_name[5] if model_name in old_models else model_name[0]
        z = int(model_name.split("-")[-1])

        for s1, s2 in styles_names(model_name):
            transferred_path = get_transferred_path(s1, s2, model_name)
            sheets_path = get_sheets_path(model_name)
            eval_dir = get_eval_dir(model_name)
            transference = f'{s1}_to_{s2}'
            for mutation in mutations:
                df_audios_path = f"{eval_dir}/df_audios-{transference}-{mutation}.pkl"
                df_sheets_path = f"{eval_dir}/df_sheets-{transference}-{mutation}.pkl"
                yield {
                    'name': f"{model_name}-{transference}-{mutation}",
                    'file_dep': [df_audios_path],
                    'actions': [(sheets_generation,
                                 [sheets_path, transferred_path, transference, mutation, df_audios_path, df_sheets_path,
                                  b, z])],
                    'targets': [df_sheets_path],
                    'verbosity': 2,
                    # 'uptodate': [False]
                }


def create_html(mutation, app_dir, dfs_sheets_paths, b, z):
    init(b, z)
    files = []
    for df_path in dfs_sheets_paths:
        df = load_pickle(df_path)
        transference = root_file_name(df_path.split('-')[-2])
        orig, _, dest = transference.split('_')
        make_html(df, orig, dest, app_dir, mutation)
        files.append(transference)

    make_index(mutation, app_dir, files)


def task_html():
    """Creates the HTML file where to see the sample of rolls created"""
    for model_name in models:
        b = model_name[5] if model_name in old_models else model_name[0]
        z = int(model_name.split("-")[-1])

        eval_dir = get_eval_dir(model_name)
        app_dir = f"{eval_dir}/app/"
        dfs_sheets_paths = []

        for mutation in mutations:
            for s1, s2 in styles_names(model_name):
                transference = f'{s1}_to_{s2}'
                df_sheets_path = f"{eval_dir}/df_sheets-{transference}-{mutation}.pkl"
                dfs_sheets_paths.append(df_sheets_path)
            yield {
                'name': f"{model_name}-{mutation}",
                'file_dep': dfs_sheets_paths,
                'actions': [(create_html,
                             [mutation, app_dir, dfs_sheets_paths, b, z])],
                # 'targets': [app_dir],
                # 'verbosity': 2,
                # 'uptodate': [False]
            }


# To use for debugging
if __name__ == '__main__':
    g = globals()
    # run_tasks(ModuleTaskLoader(g), {'train:4-br': 1})
    run(g)
