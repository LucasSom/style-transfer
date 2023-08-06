# from keras.saving.save import load_model
from tensorflow.python.keras.saving.save import load_model

from dodo import train
from model.embeddings.embeddings import get_reconstruction
from utils.files_utils import load_pickle, data_path, save_pickle, get_reconstruction_path, get_model_paths, \
    preprocessed_data_path, oversample_path, preprocessed_data_dir


def test_reconstruction():
    model_name = "4-br"
    samples = 5
    inplace = False
    verbose = False
    try:
        df_reconstructed = load_pickle(file_name=f"{data_path}embeddings/{model_name}-reconsX", verbose=verbose)
    except:
        model_path, vae_dir, _ = get_model_paths(model_name)
        model = load_model(vae_dir)
        df = load_pickle(preprocessed_data_path(4, False, False))
        df_reconstructed = get_reconstruction(df, model, model_name, 500, inplace=inplace)
        save_pickle(df_reconstructed, get_reconstruction_path(model_name))

    display_reconstruction(df_reconstructed, samples=samples)


def display_reconstruction(df, samples=5):
    for i, r in df.head(samples).iterrows():
        print(r.roll.song.name)
        print("Original:")
        r.roll.display_score()
        print("Reconstrucción:")
        r.EmbeddingRoll.display_score()
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
