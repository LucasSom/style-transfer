# from dodo import train, preprocessed_data, models
# from model.colab_tension_vae.params import init
#
#
# def train_4_bars():
#     model_name = models[0]
#     b = model_name[0]
#     init(b)
#
#     train(preprocessed_data(b), model_name)
#     return 0
#
#
# if __name__ == "__main__":
#     train_4_bars()
import os.path

import pandas as pd

from utils.files_utils import data_path, file_extension, load_pickle, save_pickle


def rename_columns_of_old_pkl(renames: dict):
    d = os.path.join(data_path, "embeddings/brmf_4b")
    dirs = [os.path.join(d, f) for f in os.listdir(d)]
    for pkl in dirs:
        print("==========================================")
        if file_extension(pkl) == ".pkl":
            print("Cargando", pkl)
            df = load_pickle(pkl)
            print(pkl, "cargado")
            if type(df) == pd.DataFrame:
                df.rename(columns=renames, inplace=True)
                print("Renombrado")
                save_pickle(df, pkl)
                print("Guardado")


# rename_columns_of_old_pkl({'Autor': 'Style', 'Titulo': 'Title'})
rename_columns_of_old_pkl({
    'Mutacion_add_sub-Bach2Frescobaldi': 'NewRoll',
    'Mutacion_add_sub-Frescobaldi2Bach': 'NewRoll',
    'Mutacion_add_sub-Bach2ragtime': 'NewRoll',
    'Mutacion_add_sub-ragtime2Bach': 'NewRoll',
    'Mutacion_add_sub-Bach2Mozart': 'NewRoll',
    'Mutacion_add_sub-Mozart2Bach': 'NewRoll',

    'Mutacion_add_sub-ragtime2Frescobaldi': 'NewRoll',
    'Mutacion_add_sub-Frescobaldi2ragtime': 'NewRoll',
    'Mutacion_add_sub-Mozart2Frescobaldi': 'NewRoll',
    'Mutacion_add_sub-Frescobaldi2Mozart': 'NewRoll',

    'Mutacion_add_sub-ragtime2Mozart': 'NewRoll',
    'Mutacion_add_sub-Mozart2ragtime': 'NewRoll',

})

