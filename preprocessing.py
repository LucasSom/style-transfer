import os
from typing import List, Dict

import dfply
import pandas as pd

from model.colab_tension_vae import util
from model.colab_tension_vae.preprocess_midi import preprocess_midi_wrapper
from roll.roll import Roll
from roll.song import Song


@dfply.make_symbolic
def df_roll_to_pm(matrices, pms):
    return [util.roll_to_pretty_midi(m, pm) for m, pm in zip(matrices, pms)]


def preprocess_data(songs: Dict[str, List[str]], compases=8) -> pd.DataFrame:
    data = [{'Autor': key,
             'Titulo': os.path.basename(path),
             'rollID': idx,
             'roll': Roll(matrix, compases=compases),
             # TODO: asignarle la canción correspondiente: guardo canciones o rolls? Checkear con March
             'oldPM': old_pm,
             'bars_skipped': bars_skipped
             }
            for key, paths in songs.items()
            for path in paths
            for old_roll, _, old_pm, bars_skipped in [preprocess_midi_wrapper(path)]
            for idx, matrix in enumerate(old_roll[:])
            ]

    # TODO: guardo canciones o rolls? Checkear con March
    dataSongs = [{'Autor': key,
                  'Titulo': os.path.basename(path),
                  'song': Song(midi_file=path, nombre=os.path.basename(path), compases=compases),
                  }
                 for key, paths in songs.items()
                 for path in paths
                 ]

    return pd.DataFrame(data)
    # return (pd.DataFrame(data)
    #         >> dfply.mutate(midi=df_roll_to_pm(dfply.X['roll'], dfply.X['oldPM']))
    #         ) # TODO: esto vuela porque al pm ya lo tengo en la canción/roll

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
