import os
from typing import List, Dict

import dfply
import pandas as pd

from model.colab_tension_vae import util
from roll.song import Song


@dfply.make_symbolic
def df_roll_to_pm(matrices, pms):
    return [util.roll_to_pretty_midi(m, pm) for m, pm in zip(matrices, pms)]


def preprocess_data(songs: Dict[str, List[str]], compases=8) -> pd.DataFrame:
    data = [{'Autor': key,
             'Titulo': os.path.basename(path),
             'roll': roll,
             }
            for key, paths in songs.items()
            for path in paths
            for roll in Song(midi_file=path, nombre=os.path.basename(path), compases=compases).rolls
            ]

    return pd.DataFrame(data)
