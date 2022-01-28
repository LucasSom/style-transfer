import copy

import dfply
import music21 as m21
import numpy as np
from IPython.core.display import display, Image

from model.colab_tension_vae import util, params as guo_params
from model.embeddings import decode_embeddings


def roll_sets_to_roll(roll_sets):
    rolls = []
    for new_roll_set in roll_sets:
        new_roll = np.array([np.hstack(x) for x in zip(*new_roll_set)])
        sampled_roll = util.result_sampling(new_roll)
        sampled_roll = np.reshape(sampled_roll, (-1, sampled_roll.shape[-1]))
        rolls.append(sampled_roll)
    return rolls


def rolls_to_midis(rolls, old_pms):
    midis = []
    for sampled_roll, old_pm in zip(rolls, old_pms):
        midis.append(roll_to_midi(sampled_roll, old_pm))
    return midis


@dfply.make_symbolic
def embedding_list_to_roll_and_midi(embeddings, old_pms, vae):
    roll_set = decode_embeddings(embeddings, vae)
    while len(old_pms) < len(roll_set):
        old_pms = old_pms + old_pms

    roll = roll_sets_to_roll(roll_set)
    midi = rolls_to_midis(roll, old_pms)

    return roll, midi


def get_roll_midi_df(df_in, column='Embedding', inline=False):
    df = df_in if inline else copy.deepcopy(df_in)

    if type(column) == list:
        for c in column:
            df = get_roll_midi_df(df_in, column=c, inline=inline)
        return df

    rolls, midis = embedding_list_to_roll_and_midi(df[column], df['Old PM'])
    df[column + ' roll'] = rolls
    df[column + ' midi'] = midis

    return df


def roll_to_midi(roll, old_pm):
    return util.roll_to_pretty_midi(roll, old_pm) if type(roll) == np.ndarray else np.NaN


class Roll:

    def __init__(self, compases=4, pulso="negra", granularity="semicorchea"):
        self.compases = compases
        self.pulso = pulso
        self.granularity = granularity
        self.roll = ...
        self.old_pm = ...
        self.midi = roll_to_midi(self.roll, self.old_pm)


lily_conv = m21.converter.subConverters.ConverterLilypond()


def roll_to_score(roll):
    high_part = instrument_roll_to_part(roll[guo_params.melody_dim], roll[:guo_params.melody_dim, :],
                                        24)
    low_part = instrument_roll_to_part(roll[-1, :],
                                       roll[guo_params.melody_dim + 1:guo_params.melody_dim + 1 + guo_params.bass_dim,
                                       :], 48)
    low_part.insert(0, m21.clef.BassClef())

    full_score = m21.stream.Score([high_part, low_part])
    return full_score


def instrument_roll_to_part(rhythm_roll, pitch_roll, pitch_offset=24):
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


def display_score_from_roll(roll):
    """
    :param: roll: ndarray de shape (64,89)
    """
    _stream = roll_to_score(roll.T)
    lily = lily_conv.write(_stream, fmt='lilypond', fp='file', subformats=['png'])
    display(Image(str(lily)))


def get_scores_from_roll(roll):
    return [roll_to_score(r.T) for r in roll]
