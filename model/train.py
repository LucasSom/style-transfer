import datetime
import getopt
import os
import shutil
import sys
from typing import List, Union

import numpy as np
import pandas as pd

try:
    from keras.callbacks import ModelCheckpoint
except ImportError:
    from tensorflow.keras.callbacks import ModelCheckpoint

from tensorflow import keras

from model.colab_tension_vae import build_model, params
from utils.files_utils import load_pickle, data_path, preprocessed_data_path, get_logs_path, get_model_paths


def get_targets(ds: np.ndarray) -> List[np.ndarray]:
    i0 = ds[:, :, :74]
    i1 = ds[:, :, 74:75]
    i2 = ds[:, :, 75:-1]
    i3 = ds[:, :, -1:]
    return [i0, i1, i2, i3]


def train_model(df: Union[pd.DataFrame, str], model_name: str, final_epoch=None, ckpt=None, loss_thold=None, verbose=2):
    base_path, vae_dir, _ = get_model_paths(model_name)
    if final_epoch is None:
        final_epoch = int(input("Until how many epochs do you want to train? "))
    if ckpt is None:
        ckpt = int(input("How many epochs do you want to run until each automatic checkpoint? "))
    if loss_thold is None:
        loss_thold = float(input("With which loss do you want to stop training? "))
    if isinstance(df, str):
        df = load_pickle(file_name=df, verbose=verbose)

    if os.path.isfile(f"{vae_dir}/initial_epoch"):
        return continue_training(df=df, model_name=model_name, final_epoch=final_epoch, loss_thold=loss_thold, ckpt=ckpt,
                                 verbose=verbose)
    else:
        return train_new_model(df=df, model_name=model_name, final_epoch=final_epoch, loss_thold=loss_thold, ckpt=ckpt,
                               verbose=verbose)


def train_new_model(df: pd.DataFrame, model_name: str, final_epoch: int, loss_thold, ckpt: int = 50, verbose=2):
    if verbose: print("Training new model")
    vae = build_model.build_model()

    return train(vae, df, model_name, 0, final_epoch, ckpt, loss_thold, verbose)


def continue_training(df: pd.DataFrame, model_name: str, final_epoch: int, loss_thold, ckpt: int = 50, verbose=2):
    base_path, vae_dir, _ = get_model_paths(model_name)

    with open(f"{vae_dir}/initial_epoch", 'rt') as f:
        initial_epoch = int(f.read())
    if verbose: print(f"Continuing training from epoch {initial_epoch}")

    try:
        vae = keras.models.load_model(vae_dir, custom_objects=dict(kl_beta=build_model.kl_beta))
    except OSError:
        new_path = os.path.join(vae_dir, 'ckpt')
        print(f"Model not found in {vae_dir}. Trying {new_path}.")
        vae = keras.models.load_model(new_path, custom_objects=dict(kl_beta=build_model.kl_beta))

    return train(vae, df, model_name, initial_epoch, final_epoch + initial_epoch, ckpt, loss_thold, verbose)


def train(vae, df, model_name, initial_epoch, final_epoch, ckpt, loss_thold, verbose=2):
    print(f"Época inicial: {initial_epoch}. Época final: {final_epoch}")
    ds = np.stack([r.matrix for r in df['roll']])
    targets = get_targets(ds)

    vae_dir = get_model_paths(model_name)[1]
    log_dir = f"{get_logs_path(model_name)}/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")


    path_to_save = f"{vae_dir}/ckpt/" if ckpt > 0 else f"{vae_dir}/"
    if os.path.isdir(path_to_save):
        shutil.rmtree(path_to_save)
    else:
        os.makedirs(path_to_save)

    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    checkpoint = ModelCheckpoint(
        filepath=path_to_save,
        monitor='loss',
        verbose=verbose > 1,
        save_best_only=True,
        mode='min',
    )

    if ckpt == 0: ckpt = final_epoch
    kl_beta = 3
    learning_rate_debug = 0
    for i in range(initial_epoch, final_epoch + 1, ckpt):
        vae.get_layer('kl_beta').variables[0].assign(kl_beta)
        kl_beta += 5e-7

        callbacks = vae.fit(
            x=ds,
            y=targets,
            verbose=verbose,
            workers=8,
            initial_epoch=i,
            epochs=i + ckpt,
            callbacks=[tensorboard_callback, checkpoint]
        )

        if learning_rate_debug != vae.optimizer.learning_rate.numpy():
            print("----------------------------- CAMBIÓ LEARNING RATE -----------------------------")
            print("Learning rate:", vae.optimizer.learning_rate.numpy())
        # vae.save(path_to_save)

        with open(f'{vae_dir}/initial_epoch', 'w') as f:
            f.write(str(i + ckpt))
        print(f"Guardado initial_epoch hasta {i + ckpt}!!")

        callbacks_history = callbacks.history
        callbacks_history['epoch'] = list(np.arange(i, i + ckpt))
        assert len(callbacks_history['epoch']) == len(callbacks_history['loss'])
        callbacks_df = pd.DataFrame(callbacks_history)

        callbacks_path = f"{get_logs_path(model_name)}_{initial_epoch}.csv"
        if os.path.isfile(callbacks_path):
            prev_callbacks = pd.read_csv(callbacks_path)
        else:
            prev_callbacks = pd.DataFrame()
            with open(callbacks_path, 'w'):
                pass

        new_callbacks = pd.concat([prev_callbacks, callbacks_df])
        new_callbacks.to_csv(callbacks_path)
        print("Guardado el csv!!")

        if callbacks_history['loss'][-1] < loss_thold:
            break

    return vae


def usage():
    print(f'Usage: train.py -d --datapath <data path> (by default, {data_path})\n'
          "| -c --config <config name> (by default, '4bar')\n"
          '| -f --file <file name where to save or load the preprocessing inside of the data path>\n'
          '| -m --model <name of the model to train>\n'
          '| -e --epochs <number of epochs to train>\n'
          '| -k --checkpoints <number of epochs between each checkpoint>\n'
          '| -h --help (print this help)\n'
          '| -v --verbose (0 = silent, 1 = progress bar, 2 = one line per epoch; default: 2)\n')


if __name__ == "__main__":
    try:
        opts, args = getopt.getopt(sys.argv[1:], "hd:c:f:m:ne:k:v:",
                                   ["help",
                                    "datapath=",
                                    "config=",
                                    "file=",
                                    "model=",
                                    "epochs=",
                                    "checkpoints="
                                    "verbose="])
    except getopt.GetoptError as err:
        # print help information and exit:
        print(err)  # will print something like "option -a not recognized"
        usage()
        sys.exit(2)

    file_name = model_name = None
    epochs = checkpt = 0
    verbose = 2
    config_name = "4bar"

    for o, arg in opts:
        if o in ("-h", "--help"):
            usage()
            sys.exit()
        elif o in ["-d", "--datapath"]:
            data_path = arg
        elif o in ["-c", "--config"]:
            config_name = arg
        elif o in ("-f", "--file"):
            file_name = arg
        elif o in ("-m", "--model"):
            model_name = arg
        elif o in ("-e", "--epochs"):
            epochs = int(arg)
        elif o in ("-k", "--checkpoints"):
            checkpt = int(arg)
        elif o == "-v":
            verbose = arg

    params.init(config_name)

    if file_name is None:
        file_name = input(f"Insert path of file with the preprocessed data from {preprocessed_data_path}: ")
    if model_name is None:
        model_name = file_name
        print(f"Using default model name, ie, file name: {file_name}-{params.config.bars}")
    if epochs <= 0:
        epochs = input(f"Insert a positive number of epochs to train: ")
    if checkpt <= 0:
        checkpt = input(f"Insert a positive number of epochs between each checkpoint: ")

    songs = {folder: [song for song in os.listdir(data_path + folder)] for folder in args}

    try:
        df_preprocessed = load_pickle(file_name=f"{preprocessed_data_path}{file_name}", verbose=verbose)
    except getopt.GetoptError as err:
        print(err)
        print("The program experimented problems loading the preprocessed dataset. "
              "Try writing a name of an existing file.")
        sys.exit(1)

    train_model(df=df_preprocessed, model_name=model_name, final_epoch=epochs, ckpt=checkpt, verbose=verbose)
