import os
import sys
import getopt
from typing import List, Dict

import dfply
import pandas as pd

from model.colab_tension_vae import util
import model.colab_tension_vae.params as params
from utils.files_utils import data_path, load_pickle, save_pickle
from roll.song import Song


@dfply.make_symbolic
def df_roll_to_pm(matrices, pms):
    return [util.roll_to_pretty_midi(m, pm) for m, pm in zip(matrices, pms)]


def preprocess_data(songs_dict: Dict[str, List[str]]) -> pd.DataFrame:
    data = [{'Autor': key,
             'Titulo': os.path.basename(path),
             'roll': roll,
             }
            for key, paths in songs_dict.items()
            for path in paths
            for roll in Song(midi_file=path, nombre=os.path.basename(path)).rolls
            ]

    return pd.DataFrame(data)


def usage():
    print('Usage: preprocessing.py -d --datapath <data path> '
          '| -c --config <config name> '
          '| -f --file <file name where to save or load the preprocessing> '
          '| -h --help'
          '| <names of folders of the different data sets>')


if __name__ == "__main__":
    try:
        opts, args = getopt.getopt(sys.argv[1:], "hd:c:f:v", ["help", "datapath=", "config=", "file=", "verbose"])
    except getopt.GetoptError as err:
        # print help information and exit:
        print(err)  # will print something like "option -a not recognized"
        usage()
        sys.exit(2)

    if len(args) <= 3:
        print("Put name of config option ('4bar' or '8bar') and next, at least 2 folders with songs")
    else:
        file_name = None
        verbose = False
        config_name = "8bar"

        for o, arg in opts:
            if o == "-v":
                verbose = True
            elif o in ("-h", "--help"):
                usage()
                sys.exit()
            elif o in ["-c", "--config"]:
                config_name = arg
            elif o in ["-d", "--datapath"]:
                data_path = arg
            elif o in ("-f", "--file"):
                file_name = arg
        params.init(config_name)

        if file_name is None:
            file_name = "prep"
            print(f"Using default output file name, ie, {file_name}-{params.config.bars}")

        songs = {folder: [song for song in os.listdir(data_path + folder)] for folder in args}

        try:
            df = load_pickle(name=f"{file_name}-{params.config.bars}", path=data_path)
        except:
            df = preprocess_data(songs)
            save_pickle(df, name=f"{file_name}-{params.config.bars}", path=data_path)
