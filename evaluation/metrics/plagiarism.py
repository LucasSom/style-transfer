import numpy as np

from model.colab_tension_vae import params
from roll.guoroll import GuoRoll


def cmp_voice(v1, v2, voice, rest_value=12):
    if voice == 'melody':
        rest = params.config.melody_output_dim - 1
        padding = 0
    else:
        padding = params.config.melody_dim + params.config.melody_note_start_dim
        rest = padding + params.config.bass_output_dim - 1

    distance = 0
    for t1, t2 in zip(v1, v2):
        n1 = np.argmax(t1)
        n2 = np.argmax(t2)
        if (n1 + padding >= rest) ^ (n2 + padding >= rest):
            if n1 + padding > rest or n2 + padding > rest:
                print("Por acá no debería pasar nunca: el silencio es mayor estricto")
            distance += rest_value
        else:
            distance += abs(np.argmax(t2) - np.argmax(t1))
    return distance


def dumb_pitch_plagiarism(r1: GuoRoll, r2: GuoRoll, rest_value=12):
    bass_distance = cmp_voice(r1.get_bass(), r2.get_bass(), voice='bass', rest_value=rest_value)
    melody_distance = cmp_voice(r1.get_melody(), r2.get_melody(), voice='melody', rest_value=rest_value)

    return bass_distance, melody_distance


def dumb_rhythm_plagiarism(r1: GuoRoll, r2: GuoRoll):
    ...
