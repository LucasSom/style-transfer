import os
import music21 as m21
import pretty_midi

from utils.files_utils import root_file_name

debugging = True

eps = 0.01


def debug(df):
    df = df[df["Title"] == "sonata15-1-debug.mid"]
    if debugging:
        for k, r in df.iterrows():
            roll = r["Roll"]
            s = roll.score
            lily_conv = m21.converter.subConverters.ConverterLilypond()
            lily_conv.write(s, fp=f"debug_outputs/debug-1_{k}", fmt='lilypond', subformats=['png'])

    # Get a list of all the file paths that ends with .txt from in specified directory
    fileList = ['debug_outputs/' + f for f in os.listdir('debug_outputs/') if os.path.isfile(f)]
    # Iterate over the list of filepaths & remove each file.
    for filePath in fileList:
        try:
            if not filePath.endswith(".png"):
                os.remove(filePath)
        except:
            print("Error while deleting file : ", filePath)


def note_cmp(n1, n2):
    return n1.pitch == n2.pitch and abs((n1.end - n1.start) - (n2.end - n2.start)) < eps


def pm_cmp(pm1, pm2):
    if type(pm1) == str:
        pm1 = pretty_midi.PrettyMIDI(root_file_name(pm1) + '.mid')
    if type(pm2) == str:
        pm2 = pretty_midi.PrettyMIDI(root_file_name(pm2) + '.mid')

    if len(pm1.instruments) == len(pm2.instruments):
        for instr1, instr2 in zip(pm1.instruments, pm2.instruments):
            for n1, n2 in zip(instr1.notes, instr2.notes):
                if not note_cmp(n1, n2):
                    return False
        return True
    return False
