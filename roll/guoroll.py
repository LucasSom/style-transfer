import os
import glob
from typing import List

import music21 as m21
import numpy as np
from IPython.core.display import display, Image

from model.colab_tension_vae import util
import model.colab_tension_vae.params as params
from utils.audio_management import save_audio, lily_conv
from utils.files_utils import root_file_name, data_path

dur_to_value = 'rsqdc'


class GuoRoll:
    """
    Class that represent a fragment of $n$ bars (8 as default, 4 in Guo's work) with attributes:

    - `matrix`: matrix of $16*bars 89$
    - `bars`: number of bars per fragment
    - `song`: reference to the object `song` to which it belongs or `None` if it was obtained from the embedding
    - `score`: score obtained from the matrix
    - `midi`: Pretty MIDI obtained from the matrix

    """

    def __init__(self, matrix, name, audio_path=data_path, song=None, verbose=False):
        """
        :param matrix: matrix of `16*n x 89` with n=number of bars
        :param name: name of roll (used on the name of midi and sheet files)
        :param song: reference to the object `song` to which it belongs or `None` if it was obtained from the embedding
        """
        self.bars = params.config.bars
        self.matrix = matrix
        self.song = song
        self.name = name
        self.score = self._roll_to_score(verbose=verbose)

        if song is None:
            self.midi = self._roll_to_midi(audio_path, old_pm=None, verbose=verbose)
        else:
            self.midi = self._roll_to_midi(audio_path, old_pm=song.old_pm, verbose=verbose)
        if verbose: print(f"Created: {self.midi}")

    def _roll_to_midi(self, path, old_pm=None, verbose=False):
        return save_audio(self.name,
                          util.roll_to_pretty_midi(self.matrix, old_pm, verbose=verbose),
                          path,
                          old_pm,
                          verbose)

    def _roll_to_score(self, verbose=False):
        def instrument_roll_to_part(rhythm_roll, pitch_roll, pitch_offset=24, verbose=False):
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
                    if verbose: print(r.duration)
                t = t2
            return n_part

        high_part = instrument_roll_to_part(self.get_melody_changes().T, self.get_melody().T, 24, verbose)
        low_part = instrument_roll_to_part(self.get_bass_changes().T, self.get_bass().T, 48, verbose)
        low_part.insert(0, m21.clef.BassClef())

        full_score = m21.stream.Score([high_part, low_part])
        return full_score

    def get_melody(self):
        return self.matrix[:, :params.config.melody_dim]

    def get_melody_changes(self):
        return self.matrix[:, params.config.melody_dim]

    def get_bass(self):
        return self.matrix[:, params.config.melody_dim + 1: -1]

    def get_bass_changes(self):
        return self.matrix[:, -1]

    def get_adjacent_intervals(self, voice='melody') -> List[int]:
        def get_intervals(voice_part, changes):
            intervals = []
            prev_note = np.argmax(voice_part[0])
            for i, c in enumerate(changes[1:], start=1):
                if c:
                    new_note = np.argmax(voice_part[i])
                    if new_note == 73:  # it is a rest
                        intervals.append('rest')
                    else:
                        interval = new_note - prev_note
                        while abs(interval) > 12:  # if the interval is compound, transform it to simple
                            interval -= 12 if interval > 0 else -12
                        intervals.append(interval)
                        if intervals[-1] != 'rest':
                            prev_note = new_note
            return intervals

        if voice == 'melody':
            return get_intervals(self.get_melody(), self.get_melody_changes())
        if voice == 'bass':
            return get_intervals(self.get_bass(), self.get_bass_changes())
        # if voice == 'both':
        #     return get_intervals(self.get_melody(), self.get_melody_changes()), \
        #            get_intervals(self.get_bass(), self.get_bass_changes())

    def get_adjacent_rhythmic_patterns(self, voice='melody') -> List[str]:
        def get_rp(changes) -> List[str]:
            return [pattern_to_str(changes[i: i + 4]) for i in range(0, changes.size, 4)]

        if voice == 'melody':
            return get_rp(self.get_melody_changes())
        if voice == 'bass':
            return get_rp(self.get_bass_changes())

    def display_score(self, file_name='file', fmt='png', do_display=True):
        # file_name += f'.{fmt}'
        lily = lily_conv.write(self.score, fmt='lilypond', fp=file_name, subformats=[fmt])

        if do_display:
            display(Image(str(lily)))

        files = glob.glob(root_file_name(lily) + '*')
        for f in files:
            if not (f.endswith('png') or f.endswith('pdf')):
                os.remove(f)

        print("File saved in ", os.path.abspath(lily))
        return lily


def rolls_to_midis(rolls):
    return [sampled_roll.midi for sampled_roll in rolls]


def get_scores_from_roll(roll):
    return [r.score for r in roll]


def pattern_to_str(pattern):
    s = ""
    for i in pattern:
        s += str(int(i))
    return s


class GuoRollSmall(GuoRoll):

    def __init__(self, roll: GuoRoll):
        self.matrix = roll.matrix
        self.score = roll.score
        self.midi = None
