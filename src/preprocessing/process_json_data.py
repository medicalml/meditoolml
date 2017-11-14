import argparse
import cv2
import numpy as np
import os
import tqdm
import json

parser = argparse.ArgumentParser(description='Script for processing data in json format')
parser.add_argument('path', help='Path of json file to process')
parser.add_argument('output_path', help='Output path to save images')

args = vars(parser.parse_args())


def scale_img(img):
    img = np.array(img)
    img = img.reshape((75, 75))
    img -= img.min()
    img /= img.max()
    img *= 255
    return img


def get_file_name(id, inc_angle, is_iceberg, which_band):
    return '{}_{}_{}_{}.png'.format(id, inc_angle, is_iceberg, which_band)


def process_imgs(path, output_path):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    with open(path) as f:
        train_data = json.load(f)
    for img in tqdm.tqdm(train_data):
        band_1 = img['band_1']
        band_2 = img['band_2']
        band_1 = scale_img(band_1)
        band_2 = scale_img(band_2)

        id = img['id']
        inc_angle = img['inc_angle']
        if 'is_iceberg' in img.keys():
            is_iceberg = img['is_iceberg']
        else:
            is_iceberg = 'unknown'
        filename_band1 = get_file_name(id, inc_angle, is_iceberg, 'band1')
        filename_band2 = get_file_name(id, inc_angle, is_iceberg, 'band2')
        cv2.imwrite(os.path.join(output_path, filename_band1), band_1)
        cv2.imwrite(os.path.join(output_path, filename_band2), band_2)
        # cv2.imwrite(os.path.join(OUTPUT_PATH, filename_band1), img_data)


if __name__ == '__main__':
    process_imgs(args['path'], args['output_path'])


