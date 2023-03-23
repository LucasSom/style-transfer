import model.colab_tension_vae.build_model
import keras
vae = keras.models.load_model('data/brmf_4b/vae/ckpt/')
v = vae.get_layer('kl_beta').variables[0]
print(f'Beta value = {v} (that is, {v / 5e-7} iterations)')
