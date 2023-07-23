import getopt
import os
import sys
from collections import Counter

import p_tqdm
from typing import List, Dict

import dfply
import pandas as pd

import model.colab_tension_vae.params as params
from model.colab_tension_vae import util
from roll.song import Song
from utils import files_utils
from utils.files_utils import datasets_path, save_pickle, preprocessed_data_dir, root_file_name, \
    original_audios_path, data_path


@dfply.make_symbolic
def df_roll_to_pm(matrices, pms, verbose=False):
    return [util.roll_to_pretty_midi(m, pm, verbose=verbose) for m, pm in zip(matrices, pms)]


def preprocess_data(songs_dict: Dict[str, List[str]], save_midis: bool, sparse: bool, verbose=False) -> pd.DataFrame:
    """
    Preprocess subdatasets of midi files, creating a DataFrame of rolls prepared to use as dataset to train the model.

    :param songs_dict: Dictionary of sub-datasets. Key: name of sub-dataset. Value: name of each midi file.
    :param save_midis: whether to save midi files when building GuoRolls
    :param sparse: whether to save sparse matrices or dense matrices
    :param verbose: Whether to print intermediate messages.
    :return: DataFrame with 3 columns: 'Style', 'Title' and 'roll' (the GuoRolls of each song).
    """
    paths = [(key, os.path.basename(path), path)
             for key, paths in songs_dict.items()
             for path in paths]

    def f(author, title, path):
        song = Song(midi_file=os.path.join(datasets_path, path), nombre=os.path.basename(path),
                    audio_path=original_audios_path, save_midi=save_midis, sparse=sparse, verbose=verbose)
        return author, title, song

    rolls_list = []
    # for i, (a, t, p) in enumerate(paths):
    #     rolls_list.append(f(a, t, p))
    #     print(f"Roll: {i}/{len(paths)} - {(i/len(paths)*100):.2f}%")
    rolls_list = p_tqdm.p_map(f, *zip(*paths))
    data = [{'Style': author,
             'Title': root_file_name(title),
             'roll_id': i,
             'roll': roll,
             }
            for author, title, song in rolls_list
            for i, roll in enumerate(song.rolls)
            ]

    return pd.DataFrame(data)


def usage():
    print('Preprocess subdatasets of midi files, creating a DataFrame of rolls prepared to use as dataset to train the '
          "model. This DataFrame has 3 columns: 'Author', 'Title' and 'roll' and is saved as a pickle file.\n\n"
          '======== Usage ========\n'
          'preprocessing.py -d --datapath <data path>\n'
          "               | -c --config <config name> (by default, '4bar')\n"
          '               | -f --file <file name where to save the preprocessing inside of the data path>\n'
          '               | -h --help\n'
          '               | -v --verbose\n'
          '               | <names of folders of the different data sets>\n'
          f'default data path: {data_path}')


if __name__ == "__main__":
    try:
        opts, args = getopt.getopt(sys.argv[1:], "hd:c:f:v", ["help", "datapath=", "config=", "file=", "verbose"])
    except getopt.GetoptError as err:
        # print help information and exit:
        print(err)  # will print something like "option -a not recognized"
        usage()
        sys.exit(2)

    if ('-h', '') in opts or ('', '--help') in opts:
        usage()
        sys.exit()
    if len(args) < 2:
        print("ERROR: not enough arguments. Put name of at least 2 folders with songs")
        usage()
        sys.exit()

    file_name = None
    verbose = False
    config_name = "4bar"
    _data_path = files_utils.data_path

    for o, arg in opts:
        if o in ["-d", "--datapath"]:
            _data_path = arg
        elif o in ["-c", "--config"]:
            config_name = arg
        elif o in ("-f", "--file"):
            file_name = arg
        elif o == "-v":
            verbose = True
    params.init(config_name)

    if file_name is None:
        file_name = "prep"
        print(f"Using default output file name, ie, {preprocessed_data_dir + file_name}-{params.config.bars}")
        file_name = f"{preprocessed_data_dir + file_name}-{params.config.bars}"
    else:
        file_name = f"{file_name}-{params.config.bars}"

    if os.path.exists(file_name):
        print(f"The file {file_name} already exists. Skip it and try with other name again.")
    else:
        songs = {folder: [f"{folder}/{song}" for song in os.listdir(_data_path + folder)] for folder in args}

        df = preprocess_data(songs, False, False, verbose=verbose)
        save_pickle(df, file_name=preprocessed_data_dir + file_name, verbose=verbose)
        print("=================================================================================\n",
              f"Saved dataset preprocessed in {file_name}.pkl",
              "=================================================================================")


def oversample(df: pd.DataFrame) -> pd.DataFrame:
    """
    Oversample the minority classes
    """
    c = Counter(df["Style"])
    m = 0
    for s, v in c.items():
        if v > m:
            m = v
            s_max = s

    sample_df = pd.DataFrame()
    for s in set(df["Style"]):
        if s != s_max:
            sub_df = df[df["Style"] == s]
            n = sub_df.shape[0]
            sample_df = pd.concat([sample_df, sub_df.sample(n=m-n, random_state=41, replace=True)])
    return pd.concat([df, sample_df])
