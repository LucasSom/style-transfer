import getopt
import os
import sys
import p_tqdm
from typing import List, Dict

import dfply
import pandas as pd

import model.colab_tension_vae.params as params
from model.colab_tension_vae import util
from roll.song import Song
from utils.files_utils import datasets_path, save_pickle, preprocessed_data_path


@dfply.make_symbolic
def df_roll_to_pm(matrices, pms, verbose=False):
    return [util.roll_to_pretty_midi(m, pm, verbose=verbose) for m, pm in zip(matrices, pms)]


def preprocess_data(songs_dict: Dict[str, List[str]], verbose=False) -> pd.DataFrame:
    """
    Preprocess subdatasets of midi files, creating a DataFrame of rolls prepared to use as dataset to train the model.

    :param songs_dict: Dictionary of subdatasets. Key: name of subdataset. Value: name of each midi file.
    :param verbose: Whether to print intermediate messages.
    :return: DataFrame with 3 columns: 'Autor', 'Titulo' and 'roll' (the GuoRolls of each song).
    """
    paths = [(key, os.path.basename(path), path)
             for key, paths in songs_dict.items()
             for path in paths]

    def f(author, title, path):
        song = Song(midi_file=os.path.join(datasets_path, path), 
                    nombre=os.path.basename(path), verbose=verbose)
        return author, title, song

    rolls_list = p_tqdm.p_map(f, *zip(*paths))
    data = [{'Autor': author,
             'Titulo': title,
             'roll': roll,
             }
            for author, title, song in rolls_list
            for roll in song.rolls
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

    for o, arg in opts:
        if o in ["-d", "--datapath"]:
            data_path = arg
        elif o in ["-c", "--config"]:
            config_name = arg
        elif o in ("-f", "--file"):
            file_name = arg
        elif o == "-v":
            verbose = True
    params.init(config_name)

    if file_name is None:
        file_name = "prep"
        print(f"Using default output file name, ie, {preprocessed_data_path + file_name}-{params.config.bars}")
        file_name = f"{preprocessed_data_path + file_name}-{params.config.bars}"
    else:
        file_name = f"{file_name}-{params.config.bars}"

    if os.path.exists(file_name):
        print(f"The file {file_name} already exists. Skip it and try with other name again.")
    else:
        songs = {folder: [song for song in os.listdir(data_path + folder)] for folder in args}

        df = preprocess_data(songs, verbose=verbose)
        save_pickle(df, file_name=preprocessed_data_path+file_name, verbose=verbose)
        print("=================================================================================\n",
              f"Saved dataset preprocessed in {file_name}.pkl",
              "=================================================================================")
