# -*- coding: UTF-8 -*-
"""
@author: Jan Butora (jan.butora@univ-lille.fr)
@author: Patrick Bas (patrick.bas@univ-lille.fr)
CNRS, University of Lille 2024
"""

import numpy as np
from glob import glob
from PIL import Image
import os
from bruteforce_adobe_pattern import *
from detectAdobe import *
import shutil


def crop_image(folder, crop_size=2048):
    """Force re-cropping of images to the specified size."""
    cropped_folder = os.path.join(folder, 'cropped_images')

    # Always delete and recreate
    if os.path.exists(cropped_folder):
        shutil.rmtree(cropped_folder)
        print(f"Deleted existing cropped folder: {cropped_folder}")
    os.makedirs(cropped_folder)
    print(f"Created new cropped folder at: {cropped_folder}")

    for filename in os.listdir(folder):
        if filename.lower().endswith(".jpg"):
            image_path = os.path.join(folder, filename)
            loaded_image = Image.open(image_path)
            image = np.array(loaded_image)

            # Handle grayscale or color images
            if image.ndim == 3:
                height, width = image.shape[:2]
            else:
                height, width = image.shape

            center_x, center_y = height // 2, width // 2
            start_x = center_x - crop_size // 2
            start_y = center_y - crop_size // 2
            end_x = start_x + crop_size
            end_y = start_y + crop_size

            cropped_image = image[start_x:end_x, start_y:end_y]
            cropped_image = Image.fromarray(cropped_image)
            cropped_image.save(os.path.join(cropped_folder, filename))


def main(use_cropped=False):
    FPR = 1e-5
    w_path = './w.npy'

    if os.path.exists(w_path):
        w = np.load(w_path)
        print('Loaded existing Adobe pattern.')
    else:
        print('Estimating the Adobe pattern...')
        w = estimate_pattern()
        np.save(w_path, w)
        print('Adobe pattern estimated and saved.')

    folderpath = '/Users/charliecameron/CodingHub/CGC_Solutions/AdobeProject/adobe-detector/Data/SusClusters/Cluster1'

    if use_cropped:
        crop_image(folderpath)
        image_path_pattern = os.path.join(folderpath, 'cropped_images', '*.jpg')
    else:
        image_path_pattern = os.path.join(folderpath, '*.jpg')

    im_list = np.array(sorted(glob(image_path_pattern)))
    print(f"Found {len(im_list)} {'cropped' if use_cropped else 'original'} images.")

    for file in im_list:
        img = Image.open(file)
        print(f"{file}: {np.array(img).shape}")

    pattern_present = np.zeros((len(im_list)), dtype=bool)
    statistic = np.zeros((len(im_list)))

    for i, im_name in enumerate(im_list):
        pattern_present[i], statistic[i] = detect_adobe_pattern(im_name, w, FPR)

    print('Images:', im_list)
    print('Test statistics:', statistic)
    print('Decision threshold for FPR {} : {}'.format(FPR, get_threshold(FPR, 0, 1 / np.sqrt(w.size))))
    print('Adobe pattern detected in:')
    for idx, detected in enumerate(pattern_present):
        if detected:
            print(f"  {im_list[idx]}")


if __name__ == '__main__':
    import cProfile
    # Toggle use_cropped here
    cProfile.run('main(use_cropped=True)', 'adobe_profile.prof')