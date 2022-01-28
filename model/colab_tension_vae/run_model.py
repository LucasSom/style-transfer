import pickle
import numpy as np

import build_model


with open('data/canciones.pkl', 'rb') as f:
    songs_preprocessed = pickle.load(f)

vae = build_model.build_model()

v = np.vstack(songs_preprocessed[:4])
i0 = v[:, :, :74]
i1 = v[:, :, 74:75]
i2 = v[:, :, 75:-1]
i3 = v[:, :, -1:]
targets = [i0, i1, i2, i3, i3, i3]

vae.fit(x=v, y=targets, verbose=2, epochs=2)
