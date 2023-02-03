import os
import sys
from typing import List

from data_analysis.SongStatistics import SongStatistics


# noinspection PyShadowingNames
from model.colab_tension_vae.params import init


def show_stats(files: List):
    for file in files:
        print("======================")
        print("File: ", file)
        if len(file) > 4 and file[-4:] == ".mid":
            SongStatistics(file).print_stats()
        else:
            print("......is not a midi file")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("No path passed to evaluate")
    else:
        init()
        params = sys.argv[1:]
        for p in params:
            if os.path.isdir(p):
                for (root, dirs, files) in os.walk(p, topdown=True):
                    if root[-1] != "/":
                        root = root+"/"
                    print(f"----------- Directory: {root} -----------")
                    show_stats([root+f for f in files])
            if os.path.isfile(p):
                show_stats([p])
