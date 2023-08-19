from dodo import do_transfer, styles_names
from utils.files_utils import get_characteristics_path, get_emb_path, get_model_paths, get_transferred_path, \
    load_pickle, get_reconstruction_path


def test_transfer():
    model_name = '4-br'

    model_path, vae_dir, vae_path = get_model_paths(model_name)
    characteristics_path = get_characteristics_path(model_name)
    emb_path = get_emb_path(model_name)

    s1, s2 = styles_names(model_name)[0]
    transferred_path = get_transferred_path(s1, s2, model_name)
    do_transfer(emb_path, vae_dir, characteristics_path, transferred_path, s1, s2)

    df = load_pickle(transferred_path)

    columns = ['Style', 'Title', 'roll_id', 'roll', 'Embedding', 'Mutacion_add', 'Mutacion_add_sub', 'NewRoll']
    assert len(columns) == len(df.columns)
    for c in df.columns:
        assert c in columns


def test_transfer_mixture_model():
    b, z = 4, 96
    model_name = f'4-Lakh_Kern-{z}'

    model_name_model = f"{b}-CPFRAa-{z}"
    _, vae_dir, vae_path = get_model_paths(model_name_model)

    characteristics_path = get_characteristics_path(model_name)
    rec_path = get_reconstruction_path(model_name)

    s1, s2 = styles_names(model_name)[0]
    transferred_path = get_transferred_path(s1, s2, model_name)
    do_transfer(rec_path, vae_dir, characteristics_path, transferred_path, s1, s2, b, z)
