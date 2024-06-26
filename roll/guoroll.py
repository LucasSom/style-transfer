import glob
import os
import random
import warnings
from typing import List

from scipy.sparse import csr_matrix
import music21 as m21
import numpy as np
from IPython.core.display import display, Image

import model.colab_tension_vae.params as params
from model.colab_tension_vae import util
from utils.audio_management import save_audio, lily_conv
from utils.files_utils import root_file_name, data_path, make_dirs_if_not_exists

dur_to_value = 'rsqdc'


class GuoRoll:
    """
    Class that represent a fragment of n bars (8 as default, 4 in Guo's work) with attributes:

    - matrix: matrix of 16*bars 89
    - bars: number of bars per fragment
    - song: reference to the object `song` to which it belongs or `None` if it was obtained from the embedding
    - score: score obtained from the matrix
    - midi: Pretty MIDI obtained from the matrix

    """

    def __init__(self, matrix, name='', audio_path=data_path + 'audios', song=None, save_midi=True, save_score=False,
                 sparse=True, verbose=False):
        """
        :param matrix: matrix of `16*n x 89` with n=number of bars
        :param name: name of roll (used on the name of midi and sheet files)
        :param song: reference to the object `song` to which it belongs or `None` if it was obtained from the embedding
        """
        self.bars = params.config.bars
        self.matrix = csr_matrix(matrix) if sparse else matrix
        self.sparse = sparse
        self.song = song
        self.name = name
        self.score = self.roll_to_score(verbose=verbose) if save_score else None

        if save_midi:
            if song is None:
                self.midi = self.roll_to_audio(audio_path, old_pm=None, verbose=verbose)
            else:
                self.midi = self.roll_to_audio(audio_path, old_pm=song.old_pm, verbose=verbose)
            if verbose: print(f"Created: {self.midi}")
        else:
            self.midi = None

    def roll_to_audio(self, path, audio_name=None, old_pm=None, save_mp3=False, verbose=False):
        return save_audio(self.name if audio_name is None else audio_name,
                          util.roll_to_pretty_midi(self.matrix, old_pm, self.sparse, verbose=verbose),
                          path, save_mp3, verbose)

    def roll_to_score(self, verbose=False):
        def instrument_roll_to_part(rhythm_roll, pitch_roll, pitch_offset=24, verbose=False):
            n_part = m21.stream.Part()

            t = 0
            if self.sparse:
                try:
                    rhythm_roll = rhythm_roll.toarray()[0]
                except:
                    rhythm_roll = np.array(rhythm_roll)[0]
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
        if self.sparse:
            return self.matrix.todense()[:, params.config.melody_dim]
        return self.matrix[:, params.config.melody_dim]

    def get_bass(self):
        if self.sparse:
            return self.matrix.todense()[:, params.config.melody_dim + 1: -1]
        return self.matrix[:, params.config.melody_dim + 1: -1]

    def get_bass_changes(self):
        return self.matrix[:, -1]

    def get_adjacent_intervals(self, voice='melody') -> List[int]:
        if voice == 'melody':
            return get_intervals(self.get_melody(), self.get_melody_changes())
        if voice == 'bass':
            return get_intervals(self.get_bass(), self.get_bass_changes())

    def get_adjacent_rhythmic_patterns(self, voice='melody') -> List[str]:
        if voice == 'melody':
            return get_rhythmic_patterns(self.get_melody_changes())
        if voice == 'bass':
            return get_rhythmic_patterns(self.get_bass_changes())

    def get_score(self, verbose=False):
        if self.score is None:
            return self.roll_to_score(verbose)
        return self.score

    def generate_sheet(self, file_name, fmt='png', do_display=False, verbose=False):
        # file_name += f'.{fmt}'
        make_dirs_if_not_exists(os.path.dirname(file_name))
        lily = lily_conv.write(self.get_score(verbose=verbose), fmt='lilypond', fp=file_name, subformats=[fmt])

        if do_display:
            display(Image(str(lily)))

        files = glob.glob(root_file_name(lily) + '*')
        for f in files:
            if not (f.endswith('png') or f.endswith('pdf')):
                try:
                    os.remove(f)
                except:
                    warnings.warn(f"File not found: {f}")

        print("File saved in ", os.path.abspath(lily))
        return str(lily)

    def _get_permutation(self, changes, permutation, voice, idx_change):
        durations = []
        for i, c in enumerate(changes):
            if i == changes.shape[0] - 1:
                durations.append((c, self.matrix.shape[0] - c))
            else:
                durations.append((c, changes[i + 1] - c))

        random.shuffle(durations)

        i = 0
        for start, duration in durations:
            for j in range(i, i + duration):
                permutation[j] += voice[start]
            permutation[i][idx_change] = 1
            i += duration

        return permutation

    def permute(self, melody_changes, bass_changes):
        permutation = np.zeros(self.matrix.shape)

        voice = np.concatenate((self.get_melody(),  # matrix of 64 x 74 (in 4 bars)
                                np.zeros((self.matrix.shape[0],  # matrix of 64 x 15 (in 4 bars)
                                          self.matrix.shape[1] - self.get_melody().shape[1]))),
                               axis=1)  # result: matrix of 64 x 89
        permutation = self._get_permutation(melody_changes, permutation, voice, params.config.melody_dim)

        voice = np.concatenate((np.zeros((self.get_melody().shape[0], self.get_melody().shape[1] + 1)),
                                self.get_bass(),
                                np.zeros((self.get_bass().shape[0], 1))
                                ), axis=1)
        permutation = self._get_permutation(bass_changes, permutation, voice, -1)

        return permutation

    def get_audio(self, audio_path, audio_name=None, fmt='.mid', verbose=False) -> str:
        get_mp3 = fmt == '.mp3'
        if self.midi is None or get_mp3:
            return self.roll_to_audio(audio_path, audio_name=audio_name, old_pm=None, save_mp3=get_mp3, verbose=verbose)
        return self.midi


def get_rhythmic_patterns(changes) -> List[str]:
    return [pattern_to_str(changes[i: i + 4]) for i in range(0, changes.size, 4)]


def get_intervals(voice_part, changes):
    intervals = []
    prev_note = np.argmax(voice_part[0])
    for i, c in enumerate(changes[1:], start=1):
        if c:
            new_note = np.argmax(voice_part[i])
            if new_note != 73:
                interval = new_note - prev_note
                while abs(interval) > 12:  # if the interval is compound, transform it to simple
                    interval -= 12 if interval > 0 else -12
                intervals.append(interval)
                if intervals[-1] != 'rest':
                    prev_note = new_note
            else:  # it is a rest
                # intervals.append('rest')
                pass
    return intervals


def rolls_to_midis(rolls):
    return [sampled_roll.midi for sampled_roll in rolls]


def get_scores_from_roll(roll):
    return [r.score for r in roll]


def pattern_to_str(pattern):
    s = ""
    for i in pattern:
        s += str(int(i))
    return s


def roll_permutations(roll: GuoRoll, n: int) -> List[np.ndarray]:
    melody_changes = np.argwhere(roll.get_melody_changes() == 1)
    bass_changes = np.argwhere(roll.get_bass_changes() == 1)

    melody_changes = melody_changes.reshape(melody_changes.shape[0])
    bass_changes = bass_changes.reshape(bass_changes.shape[0])

    ps = [roll.permute(melody_changes, bass_changes) for _ in range(n)]
    return ps
