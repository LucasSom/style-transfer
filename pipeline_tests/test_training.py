from model.embeddings.embeddings import obtain_embeddings, get_embeddings_roll_df
from utils.files_utils import load_pickle, data_path, save_pickle


def test_reconstruction(df, model, model_name, samples=5, inplace=False, verbose=False):
    try:
        df_reconstructed = load_pickle(file_name=f"{data_path}embeddings/{model_name}-recons", verbose=verbose)
    except:
        df_reconstructed = get_reconstruction(df, model, inplace=inplace)
        save_pickle(df_reconstructed, f"{data_path}embeddings/{model_name}")

    display_reconstruction(df_reconstructed, samples=samples)


def get_reconstruction(df, model, inplace=False):
    df_emb = obtain_embeddings(df, model, inplace)
    get_embeddings_roll_df(df_emb, model, inplace=True)
    return df_emb


def display_reconstruction(df, samples=5):
    for i, r in df.head(samples).iterrows():
        print(r.roll.song.name)
        print("Original:")
        r.roll.display_score()
        print("Reconstrucción:")
        r.EmbeddingRoll.display_score()
        print("----------------------------------------------------------------------")
