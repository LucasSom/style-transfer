import pickle


def datasets_name(ds):
    composed_name = ""
    for name in ds.keys():
        composed_name += "_" + name
    return composed_name


def save_pickle(df, name, path='data/pickles/'):
    with open(path + name + '.pkl', 'wb') as f:
        pickle.dump(df, f)


def load_pickle(name, path='data/pickles/'):
    with open(path + name + '.pkl', 'rb') as f:
        return pickle.load(f)
