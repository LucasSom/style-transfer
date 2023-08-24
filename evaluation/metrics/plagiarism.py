from typing import List, Tuple

import numpy as np
import pandas as pd

from model.colab_tension_vae import params
from roll.guoroll import GuoRoll


# TODO: revisar porque el padding en realidad no debería estar si ya le paso solo la matriz que importa.
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


def sort_by_pitch_plagiarism(rolls: List[GuoRoll], base_roll: GuoRoll, by_distance=False, voice='melody') \
        -> List[Tuple[float, GuoRoll]]:
    distances = [(dumb_pitch_plagiarism(r, base_roll)[voice == 'melody'][by_distance], r) for r in rolls]
    return sorted(distances, key=lambda x: x[0])


def sort_by_rhythm_plagiarism(rolls: List[GuoRoll], base_roll: GuoRoll, voice='melody'):
    distances = [(dumb_rhythm_plagiarism(r, base_roll)[voice == 'melody'], r) for r in rolls]
    return sorted(distances)


def sort_by_general_plagiarism(rolls: List[GuoRoll], base_roll: GuoRoll, by_distance=False) \
        -> List[Tuple[float, GuoRoll]]:
    def get_avg(r1, r2):
        melody = dumb_pitch_plagiarism(r1, r2)[by_distance]
        rhythm = dumb_rhythm_plagiarism(r1, r2)
        return (melody[0] + melody[1] + rhythm[0] + rhythm[1]) / 4

    distances = [(get_avg(r, base_roll), r) for r in rolls]
    return sorted(distances)


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


def get_plagiarism_position(df, original_roll, transferred_roll, by_distance=False) -> (int, int, float):
    """
    Computes a plagiarism rate for each roll and makes a ranking of rolls based on that rate. The rate is calculated as
    the sum of semitones that a roll differs with another (when `by_distance` parameter is set on *True*) or only how
    many time-frames the roll is not equal to the other (when by_distance parameter is *False*).

    :param df: df_transferred
    :param original_roll: original roll
    :param transferred_roll: roll after apply transference
    :param by_distance: whether to compute plagiarism rate summarizing the distances between each time frame or to count
     only how many times the roll differs to the original

    :return: a tuple with the ranking position of the transferred roll, the number of rolls in the ranking and the
     plagiarism rate computed
    """
    rolls = list(df['roll'])
    rolls.append(transferred_roll)

    position = 0
    # sorted_rolls = sort_by_general_plagiarism(rolls, original_roll, by_distance)
    sorted_rolls = sort_by_pitch_plagiarism(rolls, original_roll, by_distance)
    for i, (_, roll) in enumerate(sorted_rolls):
        if roll.name == transferred_roll.name:
            position = i
            break
    return position, len(rolls), sorted_rolls[position][0]


def get_plagiarism_ranking_table(df, orig: str, mutation: str, by_distance=False) -> pd.DataFrame:
    """
    :param df: df_transferred
    :param orig: original style
    :param mutation: type of mutation to analyze (add or add_sub)
    :param by_distance: whether to calculate plagiarism counting every distance between notes or only the amount of
    differences
    :return: a Dataframe with columns:
            - Style
            - Title
            - roll_id
            - roll
            - Reconstruction
            - {mutation}-NewRoll
            - Differences position
            - Differences relative ranking
            - Differences rate
            - Distance position
            - Distance relative ranking
            - Distance rate;
            2 counters with proportion of winners by style (the first one computed by difference, the second one by
            distance)
    """
    kind = "Distance" if by_distance else "Differences"
    table = {"Style": [],
             "Title": [],
             "roll_id": [],
             "roll": [],
             "Reconstruction": [],
             f"{mutation}-NewRoll": [],
             f"{kind} position": [],
             f"{kind} relative ranking": [],
             f"{kind} rate": [],
             "N": []
             }

    sub_df = df[df["Style"] == orig]

    for style, title, r_id, r_orig, r_rec, r_trans in zip(
            sub_df["Style"], sub_df["Title"], sub_df['roll_id'], sub_df['roll'], sub_df['Reconstruction'], sub_df[f"{mutation}-NewRoll"]):
        table["Style"].append(style)
        table["Title"].append(title)
        table["roll_id"].append(r_id)
        table["roll"].append(r_orig)
        table["Reconstruction"].append(r_rec)
        table[f"{mutation}-NewRoll"].append(r_trans)

        position, n, rate = get_plagiarism_position(df, r_orig, r_trans, by_distance=by_distance)

        table[f"{kind} position"].append(position)
        table[f"{kind} relative ranking"].append((n-position)/n)
        table[f"{kind} rate"].append(rate)
        table["N"].append(n)

    return pd.DataFrame(table)
