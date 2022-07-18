from model.embeddings.embeddings import get_reconstruction
from utils.files_utils import load_pickle, data_path, save_pickle, get_reconstruction_path


def test_reconstruction(df, model, model_name, samples=5, inplace=False, verbose=False):
    try:
        df_reconstructed = load_pickle(file_name=f"{data_path}embeddings/{model_name}-recons", verbose=verbose)
    except:
        model.name = model_name
        df_reconstructed = get_reconstruction(df, model, inplace=inplace)
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
