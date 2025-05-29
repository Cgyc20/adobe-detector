# -*- coding: UTF-8 -*-
"""
@author: Charlie Cameron working on the examply.py given by Jan Butora (jan.butora@univ-lille.fr)
@author: Patrick Bas (patrick.bas@univ-lille.fr)
CNRS, University of Lille 2024

# This will compare test cases for adobe and non-adobe collected images.
"""
import numpy as np
from glob import glob
import os
from bruteforce_adobe_pattern import *
from detectAdobe import *


def run_detection(image_glob, w, FPR, label):
    im_list = np.array(sorted(glob(image_glob)))
    pattern_present = np.zeros((len(im_list)), dtype=bool)
    statistic = np.zeros((len(im_list)))
    for i, im_name in enumerate(im_list):
        pattern_present[i], statistic[i] = detect_adobe_pattern(im_name, w, FPR)
    detected_count = np.sum(pattern_present)
    print(f"\n--- {label} ---")
    print(f'Folder: {image_glob}')
    print(f'{detected_count} of {len(im_list)} images detected to be Adobe within the folder called {os.path.dirname(image_glob)}.')
    print('Test statistics:', statistic)
    print('Adobe pattern detected in:', im_list[pattern_present])
    return pattern_present, statistic

def main():
    FPR = 1e-5
    w_path = './w.npy'

    if os.path.exists(w_path):
        w = np.load(w_path)
    else:
        print('Estimating the Adobe pattern...')
        w = estimate_pattern()
        np.save(w_path, w)

    adobe_cropped = '/Users/charliecameron/CodingHub/CGC_Solutions/AdobeProject/adobe-detector/Data/AdobeImages/cropped_images_top_left_512/*.jpg'
    non_adobe_cropped = '/Users/charliecameron/CodingHub/CGC_Solutions/AdobeProject/adobe-detector/Data/NonAdobeImages/cropped_images_top_left_512/*.jpg'

    print('Decision threshold for FPR {} : {}'.format(FPR, get_threshold(FPR, 0, 1/np.sqrt(w.size))))

    run_detection(adobe_cropped, w, FPR, "Adobe Images")
    run_detection(non_adobe_cropped, w, FPR, "Non-Adobe Images")

if __name__ == '__main__':
    main()

