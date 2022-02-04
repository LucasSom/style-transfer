import os
from typing import List, Dict

import dfply
import pandas as pd

from model.colab_tension_vae import preprocess_midi, util


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
             'rollID': idx,
             'roll': matrix,
             'oldPM': old_pm,
             }
            for key, paths in songs.items()
            for path in paths
            for old_roll, _, old_pm in [preprocess_midi_wrapper(path)]
            for idx, matrix in enumerate(old_roll[:])
            ]

    return (pd.DataFrame(data)
            >> dfply.mutate(midi=df_roll_to_pm(dfply.X['roll'], dfply.X['oldPM']))
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
