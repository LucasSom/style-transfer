import pickle

import numpy as np
import pretty_midi

import model.colab_tension_vae.params as params
from utils.files_utils import project_path

tension_vae_dir = project_path + "/model/colab_tension_vae/"

tensile_up_feature_vector = pickle.load(open(tension_vae_dir + 'model/tensile_up_feature_vector', 'rb'))
diameter_up_feature_vector = pickle.load(open(tension_vae_dir + 'model/diameter_up_feature_vector', 'rb'))

tensile_high_feature_vector = pickle.load(open(tension_vae_dir + 'model/tensile_high_feature_vector', 'rb'))
diameter_high_feature_vector = pickle.load(open(tension_vae_dir + 'model/diameter_high_feature_vector', 'rb'))

tensile_up_down_feature_vector = pickle.load(open(tension_vae_dir + 'model/tensile_up_down_feature_vector', 'rb'))


def beat_time(pm, beat_division=4):
    beats = pm.get_beats()

    divided_beats = []
    for i in range(len(beats) - 1):
        for j in range(beat_division):
            divided_beats.append((beats[i + 1] - beats[i]) / beat_division * j + beats[i])
    divided_beats.append(beats[-1])
    down_beats = pm.get_downbeats()
    # instantes de tiempo donde empieza cada compás
    down_beat_indices = []
    for down_beat in down_beats:
        down_beat_indices.append(np.argwhere(divided_beats == down_beat)[0][0])
        # descarta todo downbeat que no esté en una subdivisión del pulso
        # en realidad no termina descartando nada, sino se rompería con el [0][0] de []

    return np.array(divided_beats), np.array(down_beat_indices)


def find_active_range(rolls, down_beat_indices, continued=False):
    """
    Keep only bars with notes. Discard bars of rest
    """
    if down_beat_indices[1] - down_beat_indices[0] == 8:
        interval = params.config.SEGMENT_BAR_LENGTH * 2
        SAMPLES_PER_BAR = 8
    elif down_beat_indices[1] - down_beat_indices[0] == 16:
        interval = params.config.SEGMENT_BAR_LENGTH
        SAMPLES_PER_BAR = 16
    else:
        return None

    track_filled = []
    for roll in rolls:
        bar_filled = []
        for bar_index in down_beat_indices:
            bar_filled.append(np.count_nonzero(roll[:, bar_index:bar_index + SAMPLES_PER_BAR]) > 0)
        track_filled.append(bar_filled)

    track_filled = np.array(track_filled)
    two_track_filled_bar = np.count_nonzero(track_filled[:2, :], axis=0) == 2
    filled_indices = []

    for i in range(0, len(two_track_filled_bar) - interval + 1, params.config.SLIDING_WINDOW):
        if continued or np.sum(two_track_filled_bar[i:i + interval]) == interval:
            filled_indices.append((i, i + interval))

    return filled_indices


def stack_data(rolls):
    melody_roll, bass_roll = rolls
    new_bass_roll = np.zeros((12, bass_roll.shape[1]))
    bass_start_roll_new = np.zeros((1, bass_roll.shape[1]))
    bass_empty_roll = np.zeros((1, bass_roll.shape[1]))

    for step in range(bass_roll.shape[1]):
        pitch = np.where(bass_roll[:, step] != 0)[0] % 12
        original_pitch = np.where(bass_roll[:, step] != 0)[0]

        if len(pitch) > 0:
            for i in pitch:
                new_pitch = i
                new_bass_roll[new_pitch, step] = 1

            # a note start
            if bass_roll[original_pitch, step] == 1:
                bass_start_roll_new[:, step] = 1
        else:
            bass_empty_roll[:, step] = 1

    new_melody_roll = np.zeros((73, melody_roll.shape[1]))
    melody_start_roll_new = np.zeros((1, melody_roll.shape[1]))
    melody_empty_roll = np.zeros((1, melody_roll.shape[1]))

    for step in range(melody_roll.shape[1]):
        pitch = np.where(melody_roll[:, step] != 0)[0]

        if len(pitch) > 0:

            original_pitch = pitch[0]
            new_pitch = pitch[0]
            shifted_pitch = new_pitch - 24

            if 0 <= shifted_pitch <= 72:
                new_melody_roll[shifted_pitch, step] = 1

                # a note start
                if melody_roll[original_pitch, step] == 1:
                    # if step > 0:
                    melody_start_roll_new[:, step] = 1

        else:
            melody_empty_roll[:, step] = 1

    concatenated_roll = np.concatenate([new_melody_roll, melody_empty_roll, melody_start_roll_new,
                                        new_bass_roll, bass_empty_roll, bass_start_roll_new])
    # concatenated_roll[:73] = altura de lo que canta la melodia
    # concatenated_roll[73] = silencio en la melodia
    # concatenated_roll[74] = nueva nota en la melodia
    # concatenated_roll[75:87] = altura de lo que canta el bajo
    # concatenated_roll[87] = silencio en el bajo
    # concatenated_roll[88] = nueva nota en el bajo
    return concatenated_roll.transpose()


def prepare_one_x(roll_concat, filled_indices, down_beat_indices, verbose=False):
    rolls = []
    bars_skipped = []
    for start, end in filled_indices:
        start_index = down_beat_indices[start]
        if end == len(down_beat_indices):
            if roll_concat[start_index:, :].shape[0] < (
                    params.config.SAMPLES_PER_BAR * params.config.SEGMENT_BAR_LENGTH):
                fill_num = (params.config.SAMPLES_PER_BAR * params.config.SEGMENT_BAR_LENGTH
                            - roll_concat[start_index:, :].shape[0])
                fill_roll = np.vstack([roll_concat[start_index:, :], np.zeros((fill_num, 89))])
            else:
                end_index = start_index + params.config.SAMPLES_PER_BAR * params.config.SEGMENT_BAR_LENGTH
                fill_roll = roll_concat[start_index:end_index]
            if fill_roll.shape[0] == (params.config.SAMPLES_PER_BAR * params.config.SEGMENT_BAR_LENGTH):
                rolls.append(fill_roll)
            else:
                if verbose: print('skip last bars')
                bars_skipped.append(fill_roll)
        else:
            end_index = down_beat_indices[end]
            # select 4 bars
            if roll_concat[start_index:end_index, :].shape[0] \
                    == (params.config.SAMPLES_PER_BAR * params.config.SEGMENT_BAR_LENGTH):
                rolls.append(roll_concat[start_index:end_index, :])
            else:
                if verbose: print('skip')
                bars_skipped.append(roll_concat[start_index:end_index, :])

    return rolls, bars_skipped


# noinspection PySimplifyBooleanCheck
def get_roll_with_continue(track_num, track, times):
    if track.notes == []:
        return np.array([[]] * 128)  # 128 = cantidad de notas distintas de un midi

    # 0 for no note, 1 for new note, 2 for continue note
    snap_ratio = 0.5

    piano_roll = np.zeros((128, len(times)))

    previous_end_step = 0
    previous_start_step = 0
    previous_pitch = 0
    for note in track.notes:

        time_step_start = np.where(note.start >= times)[0][-1]

        if note.end > times[-1]:
            time_step_stop = len(times) - 1
        else:
            time_step_stop = np.where(note.end <= times)[0][0]

        # snap note to the grid
        # snap start time step
        if time_step_stop > time_step_start:
            start_ratio = (times[time_step_start + 1] - note.start) / (
                    times[time_step_start + 1] - times[time_step_start])
            if start_ratio < snap_ratio:
                if time_step_stop - time_step_start > 1:
                    time_step_start += 1
            # snap end time step
            end_ratio = (note.end - times[time_step_stop - 1]) / (times[time_step_stop] - times[time_step_stop - 1])
            if end_ratio < snap_ratio:
                if time_step_stop - time_step_start > 1:
                    time_step_stop -= 1

        if track_num == 0:
            # melody track, ensure single melody line
            if previous_start_step > time_step_start:
                continue
            if previous_end_step == time_step_stop and previous_start_step == time_step_start:
                # si la nota se toca al mismo tiempo que la anterior y tienen la misma duración
                continue
            piano_roll[note.pitch, time_step_start] = 1
            piano_roll[note.pitch, time_step_start + 1:time_step_stop] = 2

            if time_step_start < previous_end_step:
                piano_roll[previous_pitch, time_step_start:] = 0
            previous_pitch = note.pitch
            previous_end_step = time_step_stop
            previous_start_step = time_step_start

        elif track_num == 1:
            # for bass, select the lowest pitch if the time range is the same
            if previous_end_step == time_step_stop and previous_start_step == time_step_start:
                continue
            if previous_start_step > time_step_start:
                continue
            if time_step_start < previous_end_step:
                piano_roll[previous_pitch, time_step_start:] = 0
            piano_roll[note.pitch, time_step_start] = 1
            piano_roll[note.pitch, time_step_start + 1:time_step_stop] = 2

            previous_pitch = note.pitch
            previous_end_step = time_step_stop
            previous_start_step = time_step_start
        else:
            piano_roll[note.pitch, time_step_start:time_step_stop] = 1

    return piano_roll


def get_piano_roll(pm, sample_times):
    """

    :param pm: pretty midi piano roll with at least 3 tracks
    :param sample_times:
    :return: three piano rolls
    melody mono
    bass mono
    """
    rolls = []

    for track_num in range(2):
        rolls.append(get_roll_with_continue(track_num, pm.instruments[track_num], times=sample_times))
    return rolls


def preprocess_midi(midi_file, continued=True, verbose=False):
    pm = pretty_midi.PrettyMIDI(midi_file)

    if len(pm.instruments) < 2:
        if verbose: print('track number < 2, skip')
        return

    # sixteenth_time: marca los instantes de tiempo en donde empieza una semicorchea
    # down_beat_indices: indica los índices estos tiempos donde empieza cada compás
    #   Si hay síncopa o silencio entendería que no las marca
    sixteenth_time, down_beat_indices = beat_time(pm, beat_division=int(params.config.SAMPLES_PER_BAR / 4))
    if verbose and np.diff(down_beat_indices).min != np.diff(down_beat_indices).max:
        print("Min and max from np.diff(down_beat_indices) differ:\n"
              f"{np.diff(down_beat_indices).min} != {np.diff(down_beat_indices).max}")
    matrices = get_piano_roll(pm, sixteenth_time)

    melody_matrix = matrices[0]
    bass_matrix = matrices[1]

    # if continued:
    #     filled_indices_debug = [
    #         (i, min(i + SLIDING_WINDOW, len(down_beat_indices)))
    #         for i in range(0, len(down_beat_indices), SLIDING_WINDOW)]
    #     # Queda [(0,16), (16, 32), (32,48), (48, 64), (64, 80), ...]
    #     # en lugar de [(16, 32), (48, 64), (64, 80), (96, 112), (112, 128)]
    #
    #     filled_indices = find_active_range([melody_matrix, bass_matrix], down_beat_indices, continued)
    #     print("Al final filled_indices eran iguales?", filled_indices == filled_indices_debug[:-1])
    #     print("Mi vieja propuesta:", filled_indices_debug)
    #     print("La nueva propuesta:", filled_indices)
    # else:

    # filled_indices: lista de tuplas de índices de down_beat_indices que
    #   forman las ventanas de análisis. Estas tienen un intervalo dado por
    #   SAMPLES_PER_BAR y un salto dado por SLIDING_WINDOW
    filled_indices = find_active_range([melody_matrix, bass_matrix], down_beat_indices, continued)
    # Sin tuneo tiene tamaño 7. Con tuneo, 1

    if filled_indices is None:
        if verbose: print('not enough data for melody and bass track')
        return None
    else:
        if verbose: print("Size of filled_indices:", len(filled_indices))

    # matriz dim x tiempo en fromato guo (73 melodía + silencio + ritmo,
    # 12 bajo + silencio + ritmo)
    roll_concat = stack_data([melody_matrix, bass_matrix])

    # recorta roll_concat en los frames dados por filled_indices
    x, bars_skipped = prepare_one_x(roll_concat, filled_indices, down_beat_indices, verbose=verbose)
    x = np.array(x)
    return x, filled_indices, pm, bars_skipped
    # Sin tuneo: x es un ndarray de shape 7x64x89
    # con tuneo es de 1x256x89.


def four_bar_iterate(pianoroll, model, feature_vectors,
                     factor_t,
                     factor_d,
                     first_up=True):
    number_of_iteration = pianoroll.shape[0] // 128
    result_roll = None
    tensile_strain = None
    diameter = None

    for i in range(number_of_iteration):

        random_selection = np.random.randint(0, len(feature_vectors))
        feature_vector = feature_vectors[random_selection]
        # print(f'feature vector number is {random_selection}')
        if np.array_equal(feature_vector, tensile_up_feature_vector) or \
                np.array_equal(feature_vector, tensile_up_down_feature_vector) or \
                np.array_equal(feature_vector, tensile_high_feature_vector):
            factor = factor_t
            print('tensile change')
        else:
            factor = factor_d
            print('diameter')

        for j in range(2):

            first_4_bar = 0 if j == 0 else 1
            direction = 1 if j == 0 else -1
            direction = -1 * direction if first_up is False else direction
            start_time_step = 128 * i + params.config.time_step * first_4_bar
            print(f'number_of_iteration is {i}')
            # print(f'start_time_step is {start_time_step}')
            # print(f'j is {j}')
            input_roll = np.expand_dims(pianoroll[start_time_step:start_time_step + params.config.time_step, :], 0)
            # print(f'input shape is {input_roll.shape}')
            z = model.layers[1].predict(input_roll)
            curr_factor = direction * (np.random.uniform(-1, 1) + factor)
            print(f'factor is {curr_factor}')
            z_new = z + curr_factor * feature_vector
            reconstruction_new = model.layers[2].predict(z_new)
            result_new = \
                model.colab_tension_vae.util.result_sampling(np.concatenate(list(reconstruction_new), axis=-1))[0]
            tensile_new = np.squeeze(reconstruction_new[-2])
            diameter_new = np.squeeze(reconstruction_new[-1])

            if result_roll is None:
                result_roll = result_new
                tensile_strain = tensile_new
                diameter = diameter_new
            else:
                result_roll = np.vstack([result_roll, result_new])
                tensile_strain = np.concatenate([tensile_strain, tensile_new])
                diameter = np.concatenate([diameter, diameter_new])

            # print(f'result roll shape is {result_roll.shape}')
            # print(f'tensile_strain shape is {tensile_strain.shape}')
            # print(f'diameter shape is {diameter.shape}')
            # print('\n')

    start_time_step = 128 * (number_of_iteration + 1)
    result_roll = np.vstack([result_roll, pianoroll[start_time_step:, :]])

    return result_roll, tensile_strain, diameter


if __name__ == '__main__':
    preprocess_midi('../../data/datasets/Mozart/sonata15-1-debug.mid')


def preprocess_midi_wrapper(path, verbose=False):
    pm = preprocess_midi(path, verbose=verbose)
    # print(pm)
    if pm is None:
        print(f'DEBUG: {path} preprocessing returns None')
        return [], None, None, []
    # print(f'DEBUG: {path} is not None')
    return pm
