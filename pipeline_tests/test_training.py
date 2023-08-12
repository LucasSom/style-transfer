# from keras.saving.save import load_model
import os.path

from keras.saving.legacy.save import load_model

from dodo import train
from model.colab_tension_vae.params import init
from model.embeddings.embeddings import get_reconstruction
from utils.files_utils import load_pickle, data_path, save_pickle, get_reconstruction_path, get_model_paths, \
    preprocessed_data_path, oversample_path, preprocessed_data_dir, get_emb_path


def test_reconstruction():
    model_name = "4-CPFRAa-96"
    init(4, 96)
    samples = 5
    verbose = False
    rec_path = f"{data_path}models/{model_name}/embeddings/reconstruction.pkl"
    emb_path = get_emb_path(model_name)
    try:
        df_reconstructed = load_pickle(file_name=rec_path, verbose=verbose)
    except:
        model_path, vae_dir, _ = get_model_paths(model_name)
        model = load_model(vae_dir)
        df = load_pickle(emb_path)
        df_reconstructed = get_reconstruction(df, model, model_name)
        save_pickle(df_reconstructed, get_reconstruction_path(model_name))

    assert "Reconstruction" in df_reconstructed.columns
    assert os.path.isfile(rec_path)

    display_reconstruction(df_reconstructed, samples=samples)


def display_reconstruction(df, samples=5):
    for i, row in df.head(samples).iterrows():
        print(row.roll.song.name)
        print("Original:")
        row.roll.generate_sheet(verbose=True)
        print("Reconstrucción:")
        row.Reconstruction.generate_sheet()
        print("----------------------------------------------------------------------")


def test_task_train_small():
    z = 96
    model_name = f"4-small_br-{z}"
    b = model_name[0]

    oversample_data_path = f"{preprocessed_data_dir}4-small_br.pkl"
    test_path = oversample_data_path

    train(oversample_data_path, test_path, model_name, b, z, debug=True)


def test_task_train():
    z = 96
    model_name = f"4-cplka-{z}"
    b = model_name[0]

    oversample_data_path = oversample_path(model_name)
    test_path = f"{preprocessed_data_dir}{model_name}train.pkl"

    train(oversample_data_path, test_path, model_name, b, z, debug=True)
