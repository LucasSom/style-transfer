import os
import pickle
import random

import dfply
import pandas as pd

from colab_tension_vae import preprocess_midi, util
from debug_utils import debug, debugging

# import pretty_midi
# from typing import List, Dict, Union, NamedTuple


random.seed(42)

if_train = "No"  # @param ["Si", "No"]
epoca_inicial = 5500  # @param {type:"integer"}
epoca_final = 7000  # @param {type:"integer"}

dataset_path = './data/'

dataset1 = "Bach/"  # @param {type:"string"}
dataset2 = "ragtime/"  # @param {type:"string"}
dataset3 = "Mozart/"  # @param {type:"string"}
dataset4 = "Frescobaldi/"  # @param {type:"string"}

# @title Transformar estilo
ds_original = "ragtime"  # @param ["Bach", "ragtime", "Mozart", "Frescobaldi"]
ds_objetivo = "Mozart"  # @param ["Bach", "ragtime", "Mozart", "Frescobaldi"]

nombre_pickle = ds_original + "2" + ds_objetivo


def save_pickle(df, name, path=dataset_path):
    with open(path + name + '.pkl', 'wb') as f:
        pickle.dump(df, f)


def load_pickle(name, path=dataset_path):
    with open(path + name + '.pkl', 'rb') as f:
        return pickle.load(f)


if debugging:
    songs = {"debug": ["debug/"+path for path in os.listdir(dataset_path + "debug")]}
else:
    songs = {
        # dataset1[:-1]: [dataset1+path for path in os.listdir(dataset_path+dataset1)],
        dataset2[:-1]: [dataset2 + path for path in os.listdir(dataset_path + dataset2)],
        dataset3[:-1]: [dataset3 + path for path in os.listdir(dataset_path + dataset3)],
        dataset4[:-1]: [dataset4 + path for path in os.listdir(dataset_path + dataset4)],
    }


def datasets_name(ds):
    composed_name = ""
    for name in ds.keys():
        composed_name += "_" + name
    return composed_name


### Dataset con DataFrame


def preprocess_midi_wrapper(path):
    pm = preprocess_midi.preprocess_midi(path)
    # print(pm)
    if pm is None:
        print(f'DEBUG: {path} preprocessing returns None')
        return [], None, None
    # print(f'DEBUG: {path} is not None')
    return pm


@dfply.make_symbolic
def df_roll_to_pm(matrices, pms):
    return [util.roll_to_pretty_midi(m, pm) for m, pm in zip(matrices, pms)]


def preprocess_data():
    data = [{'Autor': key,
             'Titulo': os.path.basename(path),
             'Id roll': idx,
             'Roll': matrix,
             'Old PM': old_pm,
             }
            for key, paths in songs.items()
            for path in paths
            for old_roll, _, old_pm in [preprocess_midi_wrapper(dataset_path + path)]
            for idx, matrix in enumerate(old_roll[:])
            ]

    df = pd.DataFrame(data)
    debug(df)

    (df >> dfply.group_by('Autor')
     >> dfply.summarize(count=dfply.X['Titulo'].unique().shape[0]))

    (df >> dfply.group_by('Autor', 'Titulo')
     >> dfply.summarize(count=dfply.X['Id roll'].shape[0])
     >> dfply.group_by('Autor')
     >> dfply.summarize(count_autor=dfply.X['count'].mean())
     )

    return df.groupby('Titulo').sample()
