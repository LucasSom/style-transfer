import os
import shutil
from typing import List, Union

import datetime
import numpy as np
import pandas as pd
from tensorflow import keras

from model.colab_tension_vae import build_model
from utils.files_utils import load_pickle, data_path


def get_targets(ds: np.ndarray) -> List[np.ndarray]:
    i0 = ds[:, :, :74]
    i1 = ds[:, :, 74:75]
    i2 = ds[:, :, 75:-1]
    i3 = ds[:, :, -1:]
    return [i0, i1, i2, i3]


def train_new_model(df: pd.DataFrame, model_name: str, final_epoch: int, ckpt: int = 50):
    vae = build_model.build_model()

    return train(vae, df, model_name, 0, final_epoch, ckpt)


def continue_training(df: pd.DataFrame, model_name: str, final_epoch: int, ckpt: int = 50):
    vae = keras.models.load_model(data_path + f"saved_models/{model_name}/",
                                  custom_objects=dict(kl_beta=build_model.kl_beta))

    with open(data_path + f"logs/{model_name}/initial_epoch", 'rt') as f:
        initial_epoch = int(f.read())

    return train(vae, df, model_name, initial_epoch, final_epoch + initial_epoch, ckpt)


def train_model(df: Union[pd.DataFrame, str], model_name: str, new_training: bool, final_epoch: int, ckpt: int = 50):
    if isinstance(df, str):
        df = load_pickle(name=df, path=data_path + "preprocessed_data/")
    if not os.path.isdir(data_path + f"saved_models/{model_name}"):
        os.makedirs(data_path + f"saved_models/{model_name}")

    if new_training:
        return train_new_model(df=df, model_name=model_name, final_epoch=final_epoch, ckpt=ckpt)
    else:
        return continue_training(df=df, model_name=model_name, final_epoch=final_epoch, ckpt=ckpt)


def train(vae, df, model_name, initial_epoch, final_epoch, ckpt):
    print(f"Época inicial: {initial_epoch}. Época final: {final_epoch}")
    ds = np.stack([r.matrix for r in df['roll']])
    targets = get_targets(ds)

    log_dir = data_path + f"logs/{model_name}/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    callbacks_path = data_path + f"logs/{model_name}_{initial_epoch}.csv"
    for i in range(initial_epoch, final_epoch + 1, ckpt):
        callbacks = vae.fit(
            x=ds,
            y=targets,
            verbose=2,
            workers=8,
            initial_epoch=i,
            epochs=i + ckpt,
            callbacks=[tensorboard_callback]
        )

        path_to_save = data_path + f"saved_models/{model_name}/"
        if os.path.isdir(path_to_save):
            shutil.rmtree(path_to_save)
        else:
            os.makedirs(path_to_save)

        vae.save(path_to_save)

        with open(data_path + f'logs/{model_name}/initial_epoch', 'w') as f:
            f.write(str(i + ckpt))
        print(f"Guardado hasta {i + ckpt}!!")

        callbacks_history = {k: v for k, v in callbacks.history.items() if k != "kl_loss"}

        callbacks_df = pd.DataFrame(callbacks_history)
        prev_callbacks = pd.read_csv(callbacks_path) if os.path.isfile(callbacks_path) else pd.DataFrame()
        new_callbacks = pd.concat([prev_callbacks, callbacks_df])
        new_callbacks.to_csv(callbacks_path)
        print("Guardado el csv!!")

    return vae
