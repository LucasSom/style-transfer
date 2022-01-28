import os
from typing import List, Dict

import dfply
import pandas as pd

from colab_tension_vae import preprocess_midi, util
from debug_utils import debug, debugging


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


def preprocess_data(songs: Dict[str, List[str]]) -> pd.DataFrame:
    data = [{'Autor': key,
             'Titulo': os.path.basename(path),
             'Id roll': idx,
             'Roll': matrix,
             'Old PM': old_pm,
             }
            for key, paths in songs.items()
            for path in paths
            for old_roll, _, old_pm in [preprocess_midi_wrapper(f"data/{path}")]
            for idx, matrix in enumerate(old_roll[:])
            ]

    return (pd.DataFrame(data)
            >> dfply.mutate(midi=df_roll_to_pm(dfply.X['Roll'], dfply.X['Old PM']))
            )

    # (df >> dfply.group_by('Autor')
    #  >> dfply.summarize(count=dfply.X['Titulo'].unique().shape[0]))
    #
    # (df >> dfply.group_by('Autor', 'Titulo')
    #  >> dfply.summarize(count=dfply.X['Id roll'].shape[0])
    #  >> dfply.group_by('Autor')
    #  >> dfply.summarize(count_autor=dfply.X['count'].mean())
    #  )
    #
    # return df.groupby('Titulo').sample()
