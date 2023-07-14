import datetime
import getopt
import os
import shutil
import sys
from typing import List, Union

import numpy as np
import pandas as pd
from keras import backend

from model.custom_callbacks import PrintLearningRate, IncrementKLBeta, LossHistory

try:
    from keras.callbacks import ModelCheckpoint, EarlyStopping
except ImportError:
    from tensorflow.keras.callbacks import ModelCheckpoint

from tensorflow import keras

from model.colab_tension_vae import build_model, params
from utils.files_utils import load_pickle, data_path, preprocessed_data_dir, get_logs_path, get_model_paths


def get_targets(ds: np.ndarray, sparse: bool) -> List[np.ndarray]:
    i0 = ds[:, :, :74] if not sparse else [m[:, :74] for m in ds]
    i1 = ds[:, :, 74:75] if not sparse else [m[:, 74:75] for m in ds]
    i2 = ds[:, :, 75:-1] if not sparse else [m[:, 75:-1] for m in ds]
    i3 = ds[:, :, -1:] if not sparse else [m[:, -1:] for m in ds]
    return [i0, i1, i2, i3]


def train_model(df: Union[pd.DataFrame, str], test_data, model_name: str, final_epoch=None, verbose=2, debug=False):
    base_path, vae_dir, _ = get_model_paths(model_name)
    if final_epoch is None:
        final_epoch = int(input("Until how many epochs do you want to train? ")) if not debug else 1
    if isinstance(df, str):
        df = load_pickle(file_name=df, verbose=verbose)

    if os.path.isfile(f"{vae_dir}/initial_epoch"):
        return continue_training(df=df, test_data=test_data, model_name=model_name, final_epoch=final_epoch,
                                 verbose=verbose)
    else:
        return train_new_model(df=df, test_data=test_data, model_name=model_name, final_epoch=final_epoch,
                               verbose=verbose)


def train_new_model(df: pd.DataFrame, test_data, model_name: str, final_epoch: int, verbose=2):
    if verbose: print("Training new model")
    vae = build_model.build_model()

    return train(vae, df, test_data, model_name, 0, final_epoch, verbose=verbose)


def continue_training(df: pd.DataFrame, test_data, model_name: str, final_epoch: int, verbose=2):
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

    return train(vae, df, test_data, model_name, initial_epoch, final_epoch + initial_epoch, verbose=verbose)


def train(vae, df, test_data, model_name, initial_epoch, final_epoch, batch_size=32, verbose=2):
    print(f"Época inicial: {initial_epoch}. Época final: {final_epoch}")
    sparse = list(df['roll'].head())[0].sparse

    ds = np.stack([r.matrix for r in df['roll']])
    targets = get_targets(ds, sparse)

    ds_test = np.stack([r.matrix for r in test_data['roll']])
    targets_test = get_targets(ds_test, sparse)

    vae_dir = get_model_paths(model_name)[1]
    log_dir = f"{get_logs_path(model_name)}/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    path_to_save = f"{vae_dir}/"
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
    early_stopping = EarlyStopping(monitor='val_loss', patience=3)

    initial_kl_beta = 0.0006
    kl_increment_ratio = 5e-7
    kl_threshold = 0.006
    callbacks_path = f"{get_logs_path(model_name)}_{initial_epoch}.csv"

    backend.set_value(vae.optimizer.learning_rate, 0.0001)

    def batch_generator(X, y, batch_size):
        samples_per_epoch = len(X) if type(X) is list else X.shape[0]
        number_of_batches = int(samples_per_epoch / batch_size)

        for i in range(number_of_batches):
            for b in range(i * batch_size, (i + 1) * batch_size):
                yield X[b].todense(), [y[t][b].todense() for t in range(4)]

    if sparse:
        vae.fit(
            x=batch_generator(ds, targets, batch_size),
            # y=batch_generator(targets, batch_size),
            verbose=verbose,
            workers=8,
            initial_epoch=initial_epoch,
            epochs=initial_epoch + final_epoch,
            callbacks=[tensorboard_callback,
                       checkpoint,
                       PrintLearningRate(),
                       IncrementKLBeta(initial_kl_beta, kl_increment_ratio, kl_threshold),
                       LossHistory(callbacks_path),
                       early_stopping],
            validation_data=batch_generator(ds_test, targets_test, batch_size),
            use_multiprocessing=True,
            batch_size=batch_size
        )
    else:
        vae.fit(
            x=ds,
            y=targets,
            verbose=verbose,
            workers=8,
            initial_epoch=initial_epoch,
            epochs=initial_epoch + final_epoch,
            callbacks=[tensorboard_callback,
                       checkpoint,
                       PrintLearningRate(),
                       IncrementKLBeta(initial_kl_beta, kl_increment_ratio, kl_threshold),
                       LossHistory(callbacks_path),
                       early_stopping],
            validation_data=(ds_test, targets_test),
            use_multiprocessing=True
        )

    with open(f'{vae_dir}/initial_epoch', 'w') as f:
        f.write(str(early_stopping.stopped_epoch))
    print(f"Saved initial_epoch until {early_stopping.stopped_epoch}!!")

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
        file_name = input(f"Insert path of file with the preprocessed data from {preprocessed_data_dir}: ")
    if model_name is None:
        model_name = file_name
        print(f"Using default model name, ie, file name: {file_name}-{params.config.bars}")
    if epochs <= 0:
        epochs = input(f"Insert a positive number of epochs to train: ")
    if checkpt <= 0:
        checkpt = input(f"Insert a positive number of epochs between each checkpoint: ")

    songs = {folder: [song for song in os.listdir(data_path + folder)] for folder in args}

    try:
        df_preprocessed = load_pickle(file_name=f"{preprocessed_data_dir}{file_name}", verbose=verbose)
    except getopt.GetoptError as err:
        print(err)
        print("The program experimented problems loading the preprocessed dataset. "
              "Try writing a name of an existing file.")
        sys.exit(1)

    train_model(df=df_preprocessed, test_data=test_data, model_name=model_name, final_epoch=epochs, verbose=verbose)
