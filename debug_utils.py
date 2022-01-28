import os
from roll.roll import roll_to_score
import music21 as m21

debugging = True


def debug(df):
    df = df[df["Titulo"] == "sonata15-1-debug.mid"]
    if debugging:
        for k, r in df.iterrows():
            roll = r["Roll"]
            s = roll_to_score(roll.T)
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
