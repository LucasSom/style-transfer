import music21 as m21
import numpy as np
from IPython.core.display import display, Image

from colab_tension_vae import params as guo_params



def roll_to_score(roll):
    high_part = instrument_roll_to_part(roll[guo_params.melody_dim], roll[:guo_params.melody_dim, :],
                                        guo_params.melody_dim, 24)
    low_part = instrument_roll_to_part(roll[-1, :],
                                       roll[guo_params.melody_dim + 1:guo_params.melody_dim + 1 + guo_params.bass_dim,
                                       :], guo_params.bass_dim, 48)
    low_part.insert(0, m21.clef.BassClef())

    full_score = m21.stream.Score([high_part, low_part])
    return full_score





def instrument_roll_to_part(rhythm_roll, pitch_roll, pitch_range,
                            pitch_offset=24):
    n_part = m21.stream.Part()

    t = 0
    while t < rhythm_roll.shape[0]:
        if rhythm_roll[t] == 1:
            pitch = np.nonzero(pitch_roll[:, t])[0][0]
            dur = 0
            t2 = t
            while t2 < rhythm_roll.shape[0] and pitch_roll[pitch, t2] == 1:
                dur += 1
                t2 += 1
            n = m21.note.Note(pitch_offset + pitch)
            n.quarterLength = dur / 4
            n.offset = t / 4
            n_part.append(n)
        else:
            rest = np.nonzero(pitch_roll[:, t])[0][0]
            dur = 0
            t2 = t
            while t2 < rhythm_roll.shape[0] and pitch_roll[rest, t2] == 1:
                dur += 1
                t2 += 1
            r = m21.note.Rest(duration=m21.duration.Duration(quarterLength=dur / 4))
            r.offset = t / 4
            n_part.append(r)
            print(r.duration)
        t = t2
    return n_part


def display_score(roll):
    """
  :param: roll: ndarray de shape (64,89)
  """
    _stream = roll_to_score(roll.T)
    lily = lily_conv.write(_stream, fmt='lilypond', fp='file', subformats=['png'])
    display(Image(str(lily)))


def filter_column(df, column):
    return {
        nombre: roll
        for nombre, roll in zip(df['Titulo'], df[column])
    }


def get_scores_from_roll(roll):
    return [roll_to_score(r.T) for r in roll]


