import os
import sys
import getopt
from typing import List, Dict

import dfply
import pandas as pd

from model.colab_tension_vae import util
from model.colab_tension_vae.params import configs
from utils.files_utils import data_path, load_pickle, save_pickle
from roll.song import Song


@dfply.make_symbolic
def df_roll_to_pm(matrices, pms):
    return [util.roll_to_pretty_midi(m, pm) for m, pm in zip(matrices, pms)]


def preprocess_data(songs: Dict[str, List[str]], compases) -> pd.DataFrame:
    data = [{'Autor': key,
             'Titulo': os.path.basename(path),
             'roll': roll,
             }
            for key, paths in songs.items()
            for path in paths
            for roll in Song(midi_file=path, nombre=os.path.basename(path), compases=compases).rolls
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
        config = configs["4bar"]

        for o, arg in opts:
            if o == "-v":
                verbose = True
            elif o in ("-h", "--help"):
                usage()
                sys.exit()
            elif o in ["-c", "--config"]:
                config = configs[arg]
            elif o in ["-d", "--datapath"]:
                data_path = arg
            elif o in ("-f", "--file"):
                file_name = arg

        if file_name is None:
            file_name = "prep"
            print(f"Using default output file name, ie, {file_name}-{config.bars}")

        songs = {folder: [song for song in os.listdir(data_path+folder)] for folder in args }

        try:
            df = load_pickle(name=f"{file_name}-{config.bars}", path=data_path)
        except:
            df = preprocess_data(songs, config.bars)
            save_pickle(df, name=f"{file_name}-{config.bars}", path=data_path)
