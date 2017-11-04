from src.common import Config

import numpy as np
import json
import os


def load_train_file():
    with open(os.path.join(Config.PATHS['RAW_DATA_PATH'], 'train.json')) as f:
        return json.load(f)


def load_test_file():
    with open(os.path.join(Config.PATHS['RAW_DATA_PATH'], 'test.json')) as f:
        return json.load(f)


def calculate_statistics(data_file):
    inc_angles = []
    for datapoint in data_file:
        if datapoint['inc_angle'] != 'na':
            inc_angles.append(float(datapoint['inc_angle']))

    return np.mean(inc_angles), np.std(inc_angles), (len(data_file) - len(inc_angles)) / len(data_file)


def print_stats(data_file, data_name):
    print(data_name + ': Mean: {}, std: {}, percentage of nans: {}'.format(*calculate_statistics(data_file)))


if __name__ == '__main__':
    print_stats(load_train_file(), 'train')
    print_stats(load_test_file(), 'test')
