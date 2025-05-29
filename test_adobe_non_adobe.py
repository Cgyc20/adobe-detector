# -*- coding: UTF-8 -*-
"""
@author: Charlie Cameron working on the examply.py given by Jan Butora (jan.butora@univ-lille.fr)
@author: Patrick Bas (patrick.bas@univ-lille.fr)
CNRS, University of Lille 2024

# This will compare test cases for adobe and non-adobe collected images.
"""
#import numpy as np
# from glob import glob
# import os
# from bruteforce_adobe_pattern import *
# from detectAdobe import *


# def run_detection(image_glob, w, FPR, label):
#     im_list = np.array(sorted(glob(image_glob)))
#     pattern_present = np.zeros((len(im_list)), dtype=bool)
#     statistic = np.zeros((len(im_list)))
#     for i, im_name in enumerate(im_list):
#         pattern_present[i], statistic[i] = detect_adobe_pattern(im_name, w, FPR)
#     detected_count = np.sum(pattern_present)
#     print(f"\n--- {label} ---")
#     print(f'Folder: {image_glob}')
#     print(f'{detected_count} of {len(im_list)} images detected to be Adobe within the folder called {os.path.dirname(image_glob)}.')
#     print('Test statistics:', statistic)
#     print('Adobe pattern detected in:', im_list[pattern_present])
#     return pattern_present, statistic

# def main():
#     FPR = 1e-5
#     w_path = './w.npy'

#     if os.path.exists(w_path):
#         w = np.load(w_path)
#     else:
#         print('Estimating the Adobe pattern...')
#         w = estimate_pattern()
#         np.save(w_path, w)

#     adobe_cropped = '/Users/charliecameron/CodingHub/CGC_Solutions/AdobeProject/adobe-detector/Data/AdobeImages/cropped_images_top_left_512/*.jpg'
#     non_adobe_cropped = '/Users/charliecameron/CodingHub/CGC_Solutions/AdobeProject/adobe-detector/Data/NonAdobeImages/cropped_images_top_left_512/*.jpg'

#     print('Decision threshold for FPR {} : {}'.format(FPR, get_threshold(FPR, 0, 1/np.sqrt(w.size))))

#     run_detection(adobe_cropped, w, FPR, "Adobe Images")
#     run_detection(non_adobe_cropped, w, FPR, "Non-Adobe Images")


import numpy as np
from glob import glob
import os
from scipy.stats import norm
from bruteforce_adobe_pattern import *
from detectAdobe import *


def run_detection(image_glob, w, FPR, label):
    im_list = np.array(sorted(glob(image_glob)))
    pattern_present = np.zeros((len(im_list)), dtype=bool)
    statistics = np.zeros((len(im_list)))
    best_shifts = np.zeros((len(im_list), 2), dtype=int)  # Store best shifts
    
    for i, im_name in enumerate(im_list):
        detected, best_shift, best_T, _ = detect_adobe_best_alignment(im_name, w, FPR)
        pattern_present[i] = detected
        statistics[i] = best_T
        best_shifts[i] = best_shift  # Not best_shifts = ...!
        print(f"Image: {im_name}, Detected: {detected}, Best Shift: {best_shift}, Best T: {best_T}")
    detected_count = np.sum(pattern_present)
    print(f"\n--- {label} ---")
    print(f'Folder: {image_glob}')
    print(f'{detected_count} of {len(im_list)} images detected with Adobe pattern')
    print('Test statistics:', statistics)
    print('Best shifts:', best_shifts[pattern_present])
    print('Adobe pattern detected in:', im_list[pattern_present])
    
    return pattern_present, statistics, best_shifts

def main():
    FPR = 1e-5
    w_path = './w.npy'

    if os.path.exists(w_path):
        w = np.load(w_path)
    else:
        print('Estimating the Adobe pattern...')
        w = estimate_pattern()
        np.save(w_path, w)

    # Update paths to match your system
    adobe_cropped = '/Users/charliecameron/CodingHub/CGC_Solutions/AdobeProject/adobe-detector/Data/AdobeImages/cropped_images_top_left_512/*.jpg'
    non_adobe_cropped = '/Users/charliecameron/CodingHub/CGC_Solutions/AdobeProject/adobe-detector/Data/NonAdobeImages/cropped_images_top_left_512/*.jpg'
    # Calculate threshold with Bonferroni correction
    num_tests = w.size  # Number of shifts tested (h_w × w_w)
    adjusted_alpha = FPR / num_tests
    threshold = get_threshold(adjusted_alpha, 0, 1/np.sqrt(w.size))
    
    print(f'Decision threshold for FPR {FPR} (with Bonferroni correction): {threshold:.4f}')

    # Run detection on both sets
    adobe_results, adobe_stats, adobe_shifts = run_detection(adobe_cropped, w, FPR, "Adobe Images")
    non_adobe_results, non_adobe_stats, non_adobe_shifts = run_detection(non_adobe_cropped, w, FPR, "Non-Adobe Images")

    # Optional: Save shift analysis
    np.savez('detection_results.npz',
             adobe_stats=adobe_stats,
             adobe_shifts=adobe_shifts,
             non_adobe_stats=non_adobe_stats,
             non_adobe_shifts=non_adobe_shifts)
    

if __name__ == '__main__':
    main()

