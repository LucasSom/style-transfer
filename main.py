import os

import random

random.seed(42)

try:
    from google.colab import drive

    drive.mount('/content/drive')

    # Change the directory
    os.chdir('/content/drive/MyDrive/ColabNotebooks/')
    is_colab = True
    print("Running on Colab")

except:
    is_colab = False
    print("Running locally")

dataset_path = './data/'
path_saved_models = 'model/saved_models'

# @title ¿Entrenamos un nuevo modelo? - Seleccionar directorios de los datasets
if_train = "Si"  # @param ["Si", "No"]
entrenar_nuevo = "Si"  # @param ["Si", "No"]

if entrenar_nuevo:
    epoca_inicial = 0  # @param {type:"integer"}

epoca_final = 1  # @param {type:"integer"}
checkpt = 1  # @param {type:"integer"}

dataset1 = "Bach/"  # @param {type:"string"}
dataset2 = "ragtime/"  # @param {type:"string"}
dataset3 = "Mozart/"  # @param {type:"string"}
dataset4 = "Frescobaldi/"  # @param {type:"string"}

# @title Transformar estilo
ds_original = "Bach"  # @param ["Bach", "ragtime", "Mozart", "Frescobaldi"]
ds_objetivo = "Mozart"  # @param ["Bach", "ragtime", "Mozart", "Frescobaldi"]

nombre_pickle = ds_original + "2" + ds_objetivo
