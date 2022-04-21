import os
import pickle

project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_path = project_path + '/data/'
data_tests_path = data_path + 'tests/'
preprocessed_data_path = data_path + 'preprocessed_data/'
path_saved_models = data_path + 'model/saved_models/'
logs_path = data_path + 'logs/'


def datasets_name(ds):
    composed_name = ""
    for name in ds.keys():
        composed_name += "_" + name
    return composed_name


def save_pickle(obj, name, path=data_path):
    if not os.path.isdir(path):
        os.makedirs(path)
        print("Created directory ", path)
    with open(os.path.join(path, name + ".pkl"), 'wb') as f:
        pickle.dump(obj, f)
        print("Saved as ", os.path.join(path, name + ".pkl"))


def load_pickle(name, path=data_path):
    with open(os.path.join(path, name + ".pkl"), 'rb') as f:
        return pickle.load(f)
