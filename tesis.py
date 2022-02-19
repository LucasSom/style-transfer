# import pretty_midi

# import fractions
# from typing import List, Dict, Union, NamedTuple
from IPython.display import Image

import random
import music21 as m21

from preprocessing import preprocess_data
from utils.utils import filter_column

random.seed(42)

"""#### Setup

##### Bibliotecas y constantes
"""

mxl_path = "./mxl"
path_saved_models = 'model/saved_models'
guo_path = './MIDIs_Generados/Guo/'

lily_conv = m21.converter.subConverters.ConverterLilypond()

# current_df = load_pickle('df_head')
current_df = preprocess_data()


from IPython.display import display

canciones_a_imprimir = filter_column(current_df, 'roll')
embeddings_a_imprimir = filter_column(current_df, 'Embedding')

# noised_scores = get_scores_from_roll(noised_roll)
# dup_scores = get_scores_from_roll(dup_roll)
# sigma_scores = get_scores_from_roll(sigma_roll)
# progression_scores = get_scores_from_roll(dup_progression_roll)

# all_scores = [noised_scores, dup_scores, sigma_scores, progression_scores]
scores = {nombre: roll.score
          for nombre, roll in canciones_a_imprimir.items()}

for nombre, score in scores.items():
    print(nombre)
    for i, s in enumerate(score):
        print(i)
        lily = lily_conv.write(s, fp=nombre, fmt='lilypond', subformats=['png'])
        display(Image(str(lily)))


print("Finished")
