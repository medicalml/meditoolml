from src.common import Config

import json
import os


def load_train_file():
    with open(os.path.join(Config.PATHS['RAW_DATA_PATH'], 'train.json')) as f:
        return json.load(f)


def calculate_nans():
    nb_nans = 0
    data_file = load_train_file()
    for datapoint in data_file:
        if datapoint['inc_angle'] == 'na':
            nb_nans += 1

    return nb_nans


def display_nb_nans():
    print(calculate_nans())


def display_nb_datapoints():
    print(len(load_train_file()))


if __name__ == '__main__':
    display_nb_nans()
    display_nb_datapoints()
