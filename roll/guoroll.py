import music21 as m21
import numpy as np
from IPython.core.display import display, Image

from model.colab_tension_vae import util, params as guo_params

lily_conv = m21.converter.subConverters.ConverterLilypond()


class GuoRoll:

    def __init__(self, matrix, song=None, compases=8):
        self.compases = compases
        self.matrix = matrix
        self.song = song
        self.score = self._roll_to_score()

        if song is None:
            self.midi = self._roll_to_midi(None)
        else:
            self.midi = self._roll_to_midi(song.old_pm)
        print(self.midi)

    def _roll_to_midi(self, old_pm=None):
        return util.roll_to_pretty_midi(self.matrix, old_pm)

    def _roll_to_score(self):
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

        high_part = instrument_roll_to_part(self.matrix.T[guo_params.melody_dim],
                                            self.matrix.T[:guo_params.melody_dim, :], 24)
        low_part = instrument_roll_to_part(self.matrix.T[-1, :],
                                           self.matrix.T[
                                           guo_params.melody_dim + 1: guo_params.melody_dim + 1 + guo_params.bass_dim,
                                           :], 48)
        low_part.insert(0, m21.clef.BassClef())

        full_score = m21.stream.Score([high_part, low_part])
        return full_score

    def display_score(self):
        lily = lily_conv.write(self.score, fmt='lilypond', fp='file', subformats=['png'])
        display(Image(str(lily)))


def rolls_to_midis(rolls):
    return [sampled_roll.midi for sampled_roll in rolls]


def get_scores_from_roll(roll):
    return [r.score for r in roll]
