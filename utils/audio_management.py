import os
import subprocess
import tempfile
from typing import List, Union, Tuple

import music21 as m21
import pretty_midi
from IPython.core.display import Image
from IPython.display import Audio, display

from utils.files_utils import data_path, root_file_name, get_audios_path


lily_conv = m21.converter.subConverters.ConverterLilypond()


def PlayMidi(midi_path, wav_path=None):
    if wav_path is None:
        wav_path = tempfile.mkstemp(suffix='.wav')[1]
    exec_template = 'fluidsynth -ni font.sf2 "{}" -F "{}" -r 44100'
    exec_params = exec_template.format(midi_path, wav_path)
    p = subprocess.run(exec_params, shell=True, check=True, capture_output=True)
    print(p.stdout)
    return Audio(wav_path)


def generate_audios(df, path=f"{data_path}audios/", suffix=None, verbose=0) -> Tuple[List[str], List[str]]:
    if verbose:
        print("============= Generating audios =============")

    original_rolls = df["roll"]
    original_midis = []
    for r in original_rolls:
        original_midis.append(r.roll_to_midi(path, verbose=verbose))

    new_rolls = df["NewRoll"]
    new_midis = [r.midi for r in new_rolls]
    new_titles = (df['Title'] if suffix is None
                  else df['Title'].map(lambda t: f'{root_file_name(t)}_{suffix}'))
    return save_audios(df['Title'], original_midis, path=path, verbose=verbose), \
        save_audios(new_titles, new_midis, path=path, verbose=verbose)


def save_audios(titles: List[str], midis: list, path=data_path + 'audios/', verbose=0) -> List[str]:
    """
    Generate mp3 from midis.

    :param titles: list of titles of each midi (they might have the same length).
    :param midis: list of pretty midis to convert to mp3.
    :param path: where to save the files.
    :param verbose: 0 = no verbose; 1 = only project actions; 2 = all processes.
    :return: list of names (inside path) of the mp3 files saved.
    """
    titles = [root_file_name(t) for t in titles]
    return [save_audio(name, pm, path, True, verbose) for name, pm in zip(titles, midis) if pm is not None]


def save_audio(name: str, pm: Union[str, pretty_midi.PrettyMIDI], path: str, save_mp3=True, verbose=0):
    if not os.path.exists(path):
        os.makedirs(path)
    fluids_cmd = f"fluidsynth {'-v' if verbose == 2 else ''} -a alsa -T raw -F - /usr/share/sounds/sf2/FluidR3_GM.sf2"
    ffmpeg_cmd = f"ffmpeg -y -loglevel {'info' if verbose == 2 else 'quiet'} -f s32le -i -"

    if type(pm) == str:
        pm = pretty_midi.PrettyMIDI(root_file_name(pm) + '.mid')

    file_name = os.path.join(path, name)
    pm.write(f'{file_name}.mid')
    # we convert the created midi to mp3 reading with fluidsynth and bringing it to ffmpeg
    if save_mp3:
        os.system(f"{fluids_cmd} {file_name}.mid | {ffmpeg_cmd} {file_name}.mp3")

    if verbose:
        print(f"Created {file_name}.mp3")
    return f'{file_name}.mp3'


def display_audio(song, fmt=None):
    """
    :param song: name of file (if fmt is not None, extension is ignored).
    :param fmt: format of files to use as file extension.
    """
    if fmt is not None:
        song = f'{root_file_name(song)}.{fmt}'
    audio = PlayMidi(song)
    # print(f"Listen the fabulous {s}:")
    display(audio)


def display_results(song_name, model_name, orig, target, fmt=None):
    def _display_score(song):
        lily = lily_conv.write(song, fmt='lilypond', fp='file', subformats=['png'])
        display(Image(str(lily)))

    dir_path = os.path.join(get_audios_path(model_name))
    orig_song = os.path.join(dir_path, song_name) + '_orig'
    reconstructed_song = os.path.join(dir_path, song_name) + '_recon'
    transformed_song = os.path.join(dir_path, song_name) + f'_{orig}_to_{target}'
    results = {'original': orig_song,
               'reconstrucción': reconstructed_song,
               'transformado': transformed_song,
               }

    for t, s in results.items():
        print(f"{song_name} {t}:")
        display_audio(s, fmt=fmt)
        _display_score(s)
