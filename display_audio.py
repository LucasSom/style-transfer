import os
import subprocess
import tempfile

from IPython.core.display import Image
from IPython.display import Audio, display

from roll.roll import lily_conv


def PlayMidi(midi_path, wav_path=None):
    if wav_path is None:
        wav_path = tempfile.mkstemp(suffix='.wav')[1]
    exec_template = 'fluidsynth -ni font.sf2 "{}" -F "{}" -r 44100'
    exec_params = exec_template.format(midi_path, wav_path)
    p = subprocess.run(exec_params, shell=True, check=True, capture_output=True)
    print(p.stdout)
    return Audio(wav_path)


# PlayMidi('/tmp/music21/tmp83sbvwxi.mid')

# midis = list(zip(df_caracteristicos['Titulo'],df_caracteristicos['Id matriz'],df_caracteristicos['Embedding midis']))
def save_audio(midis, path='./Evaluación/files/'):
    for i, (nombre, id, pm, pm_original) in enumerate(midis):
        file_name = path + f'{nombre}_{id}'

        pm.write(file_name + '.mid')
        pm_original.write(file_name + '_original.mid')

        # convertimos el midi creado a mp3 leyendo con fluidsynth pasándoselo a ffmpeg
        fluidsynth_command = "fluidsynth -a alsa -T raw -F - /usr/share/sounds/sf2/FluidR3_GM.sf2"
        ffmpeg_command = "ffmpeg -y -loglevel quiet -f s32le -i -"
        os.system(f"{fluidsynth_command} {file_name}.mid | {ffmpeg_command} {file_name}.mp3")
        os.system(f"{fluidsynth_command} {file_name}_original.mid | {ffmpeg_command} {file_name}_original.mp3")


def display_audio(midis, path='Evaluación/files/'):
    for nombre, id, _, _ in midis:
        audio_orig = PlayMidi(path + f'{nombre}_{id}_original.mid')
        audio = PlayMidi(path + f'{nombre}_{id}.mid')
        display(audio)
        print('Original:')
        display(audio_orig)


def display_score(s):
    lily = lily_conv.write(s, fmt='lilypond', fp='file', subformats=['png'])
    display(Image(str(lily)))
