import time
import numpy as np
from glob import glob
import os
from bruteforce_adobe_pattern import *
from detectAdobe import *

def run_detection_on_path(image_glob_path, w, FPR):
    im_list = np.array(sorted(glob(image_glob_path)))
    pattern_present = np.zeros((len(im_list)), dtype=bool)
    statistic = np.zeros((len(im_list)))

    start_time = time.time()
    for i, im_name in enumerate(im_list):
        pattern_present[i], statistic[i] = detect_adobe_pattern(im_name, w, FPR)
    elapsed = time.time() - start_time

    return im_list, pattern_present, statistic, elapsed

def main():
    FPR = 1e-5
    w_path = './w.npy'

    if os.path.exists(w_path):
        w = np.load(w_path)
    else:
        print('Estimating the Adobe pattern...')
        w = estimate_pattern()
        np.save(w_path, w)

    original_path = '/Users/charliecameron/CodingHub/CGC_Solutions/AdobeProject/adobe-detector/Data/SusClusters/Cluster1/*.jpg'
    cropped_path = '/Users/charliecameron/CodingHub/CGC_Solutions/AdobeProject/adobe-detector/Data/SusClusters/Cluster1/cropped_images_top_left/*.jpg'

    # Run detection on original images
    im_list_orig, pattern_present_orig, statistic_orig, time_orig = run_detection_on_path(original_path, w, FPR)

    # Run detection on cropped images
    im_list_crop, pattern_present_crop, statistic_crop, time_crop = run_detection_on_path(cropped_path, w, FPR)

    threshold = get_threshold(FPR, 0, 1/np.sqrt(w.size))

    print(f"Original images processed: {len(im_list_orig)}")
    print(f"Time for original images: {time_orig:.4f} seconds")
    print('Adobe pattern detected in original images:', im_list_orig[pattern_present_orig])
    print('\nTest statistics on original images:', statistic_orig)
    print('Decision threshold for FPR {} : {}'.format(FPR, threshold))

    
    print(f"\nCropped images processed: {len(im_list_crop)}")
    print(f"Time for cropped images: {time_crop:.4f} seconds")
    print('Adobe pattern detected in cropped images:', im_list_crop[pattern_present_crop])

    print('\nTest statistics on cropped images:', statistic_crop)
    print('Decision threshold for FPR {} : {}'.format(FPR, threshold))


if __name__ == '__main__':
    main()