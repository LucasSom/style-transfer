import numpy as np
import pandas as pd
from tensorflow import keras

from model.colab_tension_vae import build_model


def train_model(df: pd.DataFrame, if_train: bool, entrenar_nuevo: bool, epoca_final: int, checkpt: int = 50):
    ds = np.stack(list(df['Roll']))

    # Carga del modelo
    if if_train == "No" or entrenar_nuevo == "No":
        vae = keras.models.load_model("saved_models/")
    if if_train == "Si" and entrenar_nuevo == "Si":
        vae = build_model.build_model()

    i0 = ds[:, :, :74]
    i1 = ds[:, :, 74:75]
    i2 = ds[:, :, 75:-1]
    i3 = ds[:, :, -1:]
    targets = [i0, i1, i2, i3]

    import datetime
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    if not entrenar_nuevo:
        with open("logs/initial_epoch", 'rt') as f:
            epoca_inicial = int(f.read())

    callbacks_history = {}
    for i in range(epoca_inicial, epoca_final + 1, checkpt):
        callbacks = vae.fit(
            x=ds,
            y=targets,
            verbose=2,
            workers=8,
            initial_epoch=i,
            epochs=i + checkpt,
            callbacks=[tensorboard_callback]
        )

        vae.save("saved_models/")

        with open('logs/initial_epoch', 'w') as f:
            f.write(str(i + checkpt))
        print(f"Guardado hasta {i + checkpt}!!")

        for k, v in callbacks.history.items():
            if 'loss' in k:
                if k in callbacks_history:
                    callbacks_history[k].extend(v)
                else:
                    callbacks_history[k] = v

    return vae, callbacks_history
