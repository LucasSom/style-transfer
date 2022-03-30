import getopt
import os
import sys
from typing import List, Dict

import dfply
import pandas as pd

import model.colab_tension_vae.params as params
from model.colab_tension_vae import util
from roll.song import Song
from utils.files_utils import data_path, save_pickle


@dfply.make_symbolic
def df_roll_to_pm(matrices, pms, verbose=False):
    return [util.roll_to_pretty_midi(m, pm, verbose=verbose) for m, pm in zip(matrices, pms)]


def preprocess_data(songs_dict: Dict[str, List[str]], verbose=False) -> pd.DataFrame:
    data = [{'Autor': key,
             'Titulo': os.path.basename(path),
             'roll': roll,
             }
            for key, paths in songs_dict.items()
            for path in paths
            for roll in Song(midi_file=path, nombre=os.path.basename(path), verbose=verbose).rolls
            ]

    return pd.DataFrame(data)


def usage():
    print(f'Usage: preprocessing.py -d --datapath <data path> (by default, {data_path})\n'
          "| -c --config <config name> (by default, '4bar')\n"
          '| -f --file <file name where to save the preprocessing inside of the data path>\n'
          '| -h --help\n'
          '| -v --verbose\n'
          '| <names of folders of the different data sets>\n')


if __name__ == "__main__":
    try:
        opts, args = getopt.getopt(sys.argv[1:], "hd:c:f:v", ["help", "datapath=", "config=", "file=", "verbose"])
    except getopt.GetoptError as err:
        # print help information and exit:
        print(err)  # will print something like "option -a not recognized"
        usage()
        sys.exit(2)

    if len(args) < 2:
        print("Put name of at least 2 folders with songs")
    else:
        file_name = None
        verbose = False
        config_name = "4bar"

        for o, arg in opts:
            if o in ("-h", "--help"):
                usage()
                sys.exit()
            elif o in ["-d", "--datapath"]:
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
            print(f"Using default output file name, ie, {file_name}-{params.config.bars}")

        songs = {folder: [song for song in os.listdir(data_path + folder)] for folder in args}

        df = preprocess_data(songs, verbose=verbose)
        save_pickle(df, name=f"{file_name}-{params.config.bars}", path=data_path)
