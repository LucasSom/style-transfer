"""
This script is not included in the pipeline because is for specific purposes of the Lakh Midi Dataset only.
Run it if you want to parallelize preprocessing (after being run Guo's Midi Miner)
"""

import os

from utils.files_utils import datasets_path

if __name__ == '__main__':
    n = 16  # amount of folders where the midis will be split
    input_path = os.path.join(datasets_path, 'lmd_separated_tracks')
    ls = os.listdir(input_path)
    subdatasets = []

    for i in range(n):
        first = int(i*len(ls)/n)
        last = int(min((i+1)*len(ls)/n, len(ls)))
        subdatasets.append(ls[first:last])

    for i, subdataset in enumerate(subdatasets):
        output_path = os.path.join(datasets_path, 'sub_lmd', str(i))
        os.makedirs(output_path)
        print(output_path)
        for file_name in subdataset:
            os.replace(os.path.join(input_path, file_name), os.path.join(output_path, file_name))
