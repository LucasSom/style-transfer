import os
import shutil
from typing import List, Union

import datetime
import numpy as np
import pandas as pd
from tensorflow import keras

from model.colab_tension_vae import build_model
from utils.files_utils import load_pickle


def get_targets(ds: np.ndarray) -> List[np.ndarray]:
    i0 = ds[:, :, :74]
    i1 = ds[:, :, 74:75]
    i2 = ds[:, :, 75:-1]
    i3 = ds[:, :, -1:]
    return [i0, i1, i2, i3]


def train_new_model(df: Union[pd.DataFrame, str], model_name: str, final_epoch: int, ckpt: int = 50):
    if isinstance(df, str):
        df = load_pickle(name=df, path=f"../data/preprocessed_data/")

    vae = build_model.build_model()

    return train(vae, df, model_name, 0, final_epoch, ckpt)


def train_model(df: Union[pd.DataFrame, str], model_name: str, final_epoch: int, ckpt: int = 50):
    if isinstance(df, str):
        df = load_pickle(name=df, path=f"../data/preprocessed_data/")

    vae = keras.models.load_model(f"saved_models/{model_name}/", custom_objects=dict(kl_beta=build_model.kl_beta))

    with open(f"logs/{model_name}/initial_epoch", 'rt') as f:
        initial_epoch = int(f.read())

    return train(vae, df, model_name, initial_epoch, final_epoch, ckpt)


def train(vae, df, model_name, initial_epoch, final_epoch, ckpt):
    ds = np.stack([r.matrix for r in df['roll']])
    targets = get_targets(ds)

    log_dir = f"logs/{model_name}/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    callbacks_history = {}
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

        # if save:
        shutil.rmtree(f"saved_models/{model_name}/")
        vae.save(f"saved_models/{model_name}/")

        with open(f'logs/{model_name}/initial_epoch', 'w') as f:
            f.write(str(i + ckpt))
        print(f"Guardado hasta {i + ckpt}!!")

        for k, v in callbacks.history.items():
            if 'loss' in k and k != "kl_loss":
                if k in callbacks_history:
                    callbacks_history[k].extend(v)
                else:
                    callbacks_history[k] = v
    return vae, callbacks_history
