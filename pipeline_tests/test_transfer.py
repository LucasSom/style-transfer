import pytest

from dodo import preprocessed_data, do_embeddings, do_transfer
from utils.files_utils import load_pickle, get_embedding_path, get_characteristics_path, get_emb_path, data_path, \
    get_model_paths, get_transferred_path


def test_transfer():
    model_name = '4-br'

    model_path, vae_dir, vae_path = get_model_paths(model_name)
    characteristics_path = get_characteristics_path(model_name)
    emb_path = get_emb_path(model_name)

    for e_orig, e_dest in [("ragtime", "Bach"), ("Bach", "ragtime")]:
        transferred_path = get_transferred_path(e_orig, e_dest, model_name)
        df = do_transfer(emb_path, vae_dir, characteristics_path, e_orig, e_dest, transferred_path)

    columns = ['Style', 'Title', 'roll_id', 'roll', 'Embedding', 'Mutacion_add', 'Mutacion_add_sub', 'NewRoll']
    assert len(columns) == len(df.columns)
    for c in df.columns:
        assert c in columns