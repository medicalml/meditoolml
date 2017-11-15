import argparse
import cv2
import glob
import os
import tqdm
import numpy as np

from scipy.ndimage.filters import uniform_filter
from scipy.ndimage.measurements import variance


def lee_filter(img, size):
    img = img.astype('float32')
    img_mean = uniform_filter(img, (size, size))
    img_sqr_mean = uniform_filter(img**2, (size, size))
    img_variance = img_sqr_mean - img_mean**2

    overall_variance = variance(img)

    img_weights = img_variance**2 / (img_variance**2 + overall_variance**2)
    img_output = img_mean + img_weights * (img - img_mean)
    return img_output


def read_images_path(path):
    return glob.glob(os.path.join(path, '*'))


def parse_images(input_folder_path, output_folder_path, window_size):
    files = read_images_path(input_folder_path)
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)
    for file in tqdm.tqdm(files):
        try:
            img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
            img = img.astype('float32')
            img /= 255.
            file_name = os.path.basename(file)
            filtered = lee_filter(img, size=window_size)
            compare_img = np.zeros((75, 154))
            compare_img[:, :75] = img * 255.
            compare_img[:, 79:] = filtered * 255.

            cv2.imwrite(os.path.join(output_folder_path, file_name), compare_img)
        except Exception as e:
            print(e)


def parse_all(input_folder_path, output_folder_path, window_size):
    files = read_images_path(input_folder_path)
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)
    for file in tqdm.tqdm(files):
        try:
            img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
            img = img.astype('float32')
            img /= 255.
            file_name = os.path.basename(file)
            filtered = lee_filter(img, size=window_size)

            cv2.imwrite(os.path.join(output_folder_path, file_name), filtered * 255)
        except Exception as e:
            print(e)


parser = argparse.ArgumentParser()
parser.add_argument('path', help='Path to folder with images')
parser.add_argument('output_path', help='Path to output folder')
parser.add_argument('-w', '--window', default=7, type=float, help='Filter window size')
parser.add_argument('--parse_all', action='store_true', default=True, help='Whether use comparison with original or not')
args = parser.parse_args()

if __name__ == '__main__':
    if args.parse_all:
        parse_all(args.path, args.output_path, args.window)
    else:
        parse_images(args.path, args.output_path, args.window)
