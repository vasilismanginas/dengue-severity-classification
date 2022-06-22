import pickle
import os


def pickle_already_exists(pickle_path):
    return os.path.exists(pickle_path)


def load_pickle(pickle_path):
    pickle_name = pickle_path.split(os.path.sep)[-1]
    print(f'* Loading {pickle_name} pickle...')
    with open(pickle_path, 'rb') as this_pickle:
        data = pickle.load(this_pickle)
    print(f'* Done loading {pickle_name} pickle! \n\n')

    return data