import os
import subprocess
import tempfile
from collections import Counter
from typing import List

from IPython.core.display import Image
from IPython.display import Audio, display

from roll.guoroll import lily_conv
from utils.files_utils import data_path


def PlayMidi(midi_path, wav_path=None):
    if wav_path is None:
        wav_path = tempfile.mkstemp(suffix='.wav')[1]
    exec_template = 'fluidsynth -ni font.sf2 "{}" -F "{}" -r 44100'
    exec_params = exec_template.format(midi_path, wav_path)
    p = subprocess.run(exec_params, shell=True, check=True, capture_output=True)
    print(p.stdout)
    return Audio(wav_path)


def get_midis(df, path=f"{data_path}Audios/", column=None, suffix=None, verbose=0) -> List[str]:
    column = df.columns[-1] if column is None else column
    rolls_generated = df[column]
    if verbose:
        print("Column to generate midi:", df.columns[-1])
    midis = [r.midi for r in rolls_generated]
    titles = (df['Titulo'] if suffix is None
              else df['Titulo'].map(lambda t: f'{os.path.splitext(t)[0]}_{suffix}'))
    return save_audios(titles, midis, path=path, verbose=verbose)


# noinspection PyShadowingBuiltins
def save_audios(titles: List[str], midis: list, oldPMs: list = None, path=data_path + 'Audios/', verbose=0)\
        -> List[str]:
    """
    Generate mp3 from midis.

    :param titles: list of titles of each midi (they might have the same length).
    :param midis: list of pretty midis to convert to mp3.
    :param oldPMs: pretty midis of the original song.
    :param path: where to save the files.
    :param verbose: 0 = no verbose; 1 = only project actions; 2 = all processes.
    :return: list of names (inside path) of the mp3 files saved.
    """
    if not os.path.isdir(path):
        os.makedirs(path)

    fluidsynth_cmd = f"fluidsynth {'-v' if verbose==2 else ''} -a alsa -T raw -F - /usr/share/sounds/sf2/FluidR3_GM.sf2"
    ffmpeg_cmd = f"ffmpeg -y -loglevel {'info' if verbose==2 else 'quiet'} -f s32le -i -"

    files = []
    ids = Counter()
    titles = [os.path.splitext(t)[0] for t in titles]
    for i, (name, pm) in enumerate(zip(titles, midis)):

        ids['name'] += 1
        id = ids['name']

        file_name = os.path.join(path, f'{name}_{id}')
        pm.write(file_name + '.mid')

        # we convert the created midi to mp3 reading with fluidsynth and bringing it to ffmpeg
        os.system(f"{fluidsynth_cmd} {file_name}.mid | {ffmpeg_cmd} {file_name}.mp3")
        files.append(f'{file_name}.mp3')
        if verbose: print(f"Created {file_name}.mp3")

        if oldPMs is not None:
            pm_original = oldPMs[i]
            pm_original.write(file_name + '_original.mid')
            os.system(f"{fluidsynth_cmd} {file_name}_original.mid | {ffmpeg_cmd} {file_name}_original.mp3")
            files.append(f'{file_name}_original.mp3')
            if verbose: print(f"Created {file_name}_orginal.mp3")

    return files


# noinspection PyShadowingBuiltins
def display_audios(midi_files, path=data_path + 'Audios/'):
    """
    :param midi_files: names of midi files (if they were created with save_audios function, they would have the format
    '{name}_{id}.mp3').
    :param path: where the mp3 files are saved.
    """
    for mf in midi_files:
        audio = PlayMidi(path + mf)
        print(f"Listen the fabulous {mf}:")
        display(audio)


def display_score(s):
    lily = lily_conv.write(s, fmt='lilypond', fp='file', subformats=['png'])
    display(Image(str(lily)))
