import music21 as m21
import numpy as np
from IPython.core.display import display, Image

from model.colab_tension_vae import util
import model.colab_tension_vae.params as params

lily_conv = m21.converter.subConverters.ConverterLilypond()


class GuoRoll:
    """
    Class that represent a fragment of $n$ bars (8 as default, 4 in Guo's work) with attributes:

    - `matrix`: matrix of $16*bars 89$
    - `bars`: number of bars per fragment (es el mismo para todo el dataset, con lo cual,
    podría eliminarse la redundancia en un trabajo futuro)
    - `song`: reference to the object `song` to which it belongs or `None` if it was obtained from the embedding
    (en un trabajo futuro podría cambiárselo por un singleton).
    - `score`: score obtained from the matrix
    - `midi`: Pretty MIDI obtained from the matrix

    """

    def __init__(self, matrix, song=None):
        """
        :param matrix: matrix of `16*bars x 89` with n= la cantidad de compases
        :param song: reference to the object `song` to which it belongs or `None` if it was obtained from the embedding
        (en un trabajo futuro podría cambiárselo por un singleton).
        (es el mismo para todo el dataset, con lo cual, podría eliminarse la redundancia en un trabajo futuro)
        """
        self.bars = params.config.bars
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
                if rhythm_roll[t] == 1:  # not rest
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
                else:  # rest
                    dur = 0
                    t2 = t
                    while t2 < rhythm_roll.shape[0] and rhythm_roll[t2] == 0:
                        dur += 1
                        t2 += 1
                    r = m21.note.Rest(duration=m21.duration.Duration(quarterLength=dur / 4))
                    r.offset = t / 4
                    n_part.append(r)
                    print(r.duration)
                t = t2
            return n_part

        high_part = instrument_roll_to_part(self.matrix.T[params.config.melody_dim],
                                            self.matrix.T[:params.config.melody_dim, :], 24)
        low_part = instrument_roll_to_part(self.matrix.T[-1, :],
                                           self.matrix.T[params.config.melody_dim + 1:
                                                         params.config.melody_dim + 1 + params.config.bass_dim,
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
