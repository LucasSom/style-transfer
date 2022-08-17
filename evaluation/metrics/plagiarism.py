from typing import List

import numpy as np

from model.colab_tension_vae import params
from roll.guoroll import GuoRoll


# TODO: revisar porque el pading en realidad no debería estar si ya le paso solo la matriz que importa.
# El último elemento directamente es -1
def cmp_voice(v1, v2, voice, rest_value=12):
    if voice == 'melody':
        rest = params.config.melody_output_dim - 1
    else:
        rest = params.config.bass_output_dim - 1

    differences = 0
    distance = 0
    for t1, t2 in zip(v1, v2):
        n1 = np.argmax(t1)
        n2 = np.argmax(t2)
        if n1 > rest or n2 > rest:
            raise Exception(f"Note numbers can't be greater than the rest number. In this case, n1={n1}, n2={n2} "
                            f"and rest value is {rest}")
        if (n1 == rest) ^ (n2 == rest):
            # In some roll there is a rest but in the other there is a note
            differences += 1
            distance += rest_value
        else:
            notes_dist = abs(np.argmax(t2) - np.argmax(t1))
            if notes_dist > 0:
                differences += 1
                distance += notes_dist
    return differences, distance


def dumb_pitch_plagiarism(r1: GuoRoll, r2: GuoRoll, rest_value=12):
    bass_cmp = cmp_voice(r1.get_bass(), r2.get_bass(), voice='bass', rest_value=rest_value)
    melody_cmp = cmp_voice(r1.get_melody(), r2.get_melody(), voice='melody', rest_value=rest_value)

    return bass_cmp, melody_cmp


def dumb_rhythm_plagiarism(r1: GuoRoll, r2: GuoRoll):
    return sum(r1.get_bass_changes() ^ r2.get_bass_changes()), \
           sum(r1.get_melody_changes() ^ r2.get_melody_changes())


def sort_by_pitch_plagiarism(rolls: List[GuoRoll], base_roll: GuoRoll, by_distance=False, voice='melody'):
    distances = [(dumb_pitch_plagiarism(r, base_roll)[voice == 'melody'][by_distance], r) for r in rolls]
    return np.sort(distances)


def sort_by_rhythm_plagiarism(rolls: List[GuoRoll], base_roll: GuoRoll, voice='melody'):
    distances = [(dumb_rhythm_plagiarism(r, base_roll)[voice == 'melody'], r) for r in rolls]
    return np.sort(distances)


def sort_by_general_plagiarism(rolls: List[GuoRoll], base_roll: GuoRoll, by_distance=False):
    def get_avg(r1, r2):
        melody = dumb_pitch_plagiarism(r1, r2)[by_distance]
        rhythm = dumb_rhythm_plagiarism(r1, r2)
        return (melody[0] + melody[1] + rhythm[0] + rhythm[1]) / 4

    distances = [(get_avg(r, base_roll), r) for r in rolls]
    return np.sort(distances)


def get_most_similar_roll(base_roll: GuoRoll, rolls: List[GuoRoll], by_distance=False, voice=None, musical_element=None,
                          rest_value=12):
    d_min = 2 ^ 31
    r_min = None
    for roll in rolls:
        d = 0
        if musical_element != 'rhythm':
            if voice != 'melody':
                d += dumb_pitch_plagiarism(base_roll, roll, rest_value=rest_value)[by_distance][0]
            if voice != 'bass':
                d += dumb_pitch_plagiarism(base_roll, roll, rest_value=rest_value)[by_distance][1]
        if musical_element != 'melody':
            if voice != 'melody':
                d += dumb_rhythm_plagiarism(base_roll, roll)[0]
            if voice != 'bass':
                d += dumb_rhythm_plagiarism(base_roll, roll)[1]
        if d < d_min:
            r_min = roll
    return r_min
