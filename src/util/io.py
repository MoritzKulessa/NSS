import os
import gzip
import json
import shutil
import pickle
from tabulate import tabulate

'''
**************************************************************************************************************
**************************************************************************************************************
**************************************************************************************************************
'''


def print_pretty_table(df):
    print(tabulate(df, headers='keys', tablefmt='psql', floatfmt=".5f"))


def dump_pickle(path, file_name, data):
    if not os.path.isdir(path):
        os.makedirs(path)
    with gzip.open(path + "/" + file_name, mode="wb") as f:
        pickle.dump(data, f, protocol=4)


def load_pickle(path):
    with gzip.open(path, mode='rb') as f:
        return pickle.load(f)


def dump_json(path, file_name, data):
    if not os.path.isdir(path):
        os.makedirs(path)
    with open(path + "/" + file_name, mode="w") as f:
        json.dump(data, f)


def load_json(path):
    with open(path, mode='r') as f:
        return json.load(f)


def remove_directory(path, create=False):
    if os.path.isdir(path):
        shutil.rmtree(path)
    if create:
        os.makedirs(path)


def delete_file(path_to_file):
    os.remove(path_to_file)


def get_project_directory():
    return os.path.dirname(os.path.realpath(__file__)) + "/../../"


'''
**************************************************************************************************************
**************************************************************************************************************
**************************************************************************************************************
'''


def exist(name, loc="_results"):
    save_path = get_project_directory() + loc + "/"
    save_path += name + ".pkl"
    return os.path.exists(save_path)


def remove(name, loc="_results"):
    remove_path = get_project_directory() + loc + "/"
    remove_path += name + ".pkl"
    remove_directory(remove_path)


def save(data, name, loc="_results"):
    save_path = get_project_directory() + loc + "/"
    if not os.path.isdir(save_path): os.makedirs(save_path)
    dump_pickle(save_path, name + ".pkl", data)


def load(name, loc="_results"):
    results_path = get_project_directory() + loc + "/"
    load_path = results_path + name + ".pkl"
    return load_pickle(load_path)


def load_path(loc):
    load_path = get_project_directory() + loc
    return load_pickle(load_path)
