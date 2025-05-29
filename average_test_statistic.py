# -*- coding: UTF-8 -*-
"""
Charlie Cameron 22nd May 2025
This script evaluates the performance of the Adobe pattern detection algorithm
across images cropped to various fixed sizes from the top-left corner and compares
it with the original full images.

For each crop size (128, 256, 512, 1024, 2048 pixels), plus the original:
1. Loads the images from the corresponding folder.
2. Runs the Adobe detection algorithm on all images.
3. Records the detection test statistics, detections count, and time taken.
4. Computes and stores the average test statistic, per-image runtime, detection counts,
   success rates, and speedup relative to the original images.
5. Plots comparisons of average detection statistic, runtime per image, and speedup.

The detection uses a pre-estimated Adobe pattern (`w.npy`) and a fixed false positive rate (FPR).
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from bruteforce_adobe_pattern import *
from detectAdobe import *
import csv

def run_detection_on_path(image_glob_path, w, FPR):
    im_list = np.array(sorted(glob(image_glob_path)))
    pattern_present = np.zeros((len(im_list)), dtype=bool)
    statistic = np.zeros((len(im_list)))

    start_time = time.time()
    for i, im_name in enumerate(im_list):
        pattern_present[i], statistic[i] = detect_adobe_pattern(im_name, w, FPR)
    elapsed = time.time() - start_time

    return len(im_list), pattern_present, statistic, elapsed

def main():
    FPR = 1e-5
    base_path = '/Users/charliecameron/CodingHub/CGC_Solutions/AdobeProject/adobe-detector/Data/SusClusters/Cluster1'
    crop_sizes = [128, 256, 512, 1024, 2048]

    w_path = './w.npy'
    if os.path.exists(w_path):
        w = np.load(w_path)
    else:
        print('Estimating the Adobe pattern...')
        w = estimate_pattern()
        np.save(w_path, w)

    # First run on original full images
    original_glob = os.path.join(base_path, '*.jpg')
    print('Processing original full images...')
    n_orig, pattern_present_orig, statistic_orig, elapsed_orig = run_detection_on_path(original_glob, w, FPR)
    time_per_image_orig = elapsed_orig / n_orig if n_orig > 0 else 0

    avg_stats = []
    runtimes = []
    n_images = []
    detection_counts = []
    success_rates = []
    time_per_images = []
    speedups = []
    crop_labels = [f'{size}' for size in crop_sizes] + ['original']

    # Process cropped images
    for size in crop_sizes:
        cropped_folder = os.path.join(base_path, f'cropped_images_top_left_{size}')
        glob_path = os.path.join(cropped_folder, '*.jpg')

        print(f'\nProcessing {size}x{size} cropped images...')
        n_img, pattern_present, statistic, elapsed = run_detection_on_path(glob_path, w, FPR)

        avg_stat = np.mean(statistic) if n_img > 0 else 0
        avg_stats.append(avg_stat)
        runtimes.append(elapsed)
        n_images.append(n_img)

        count_detected = np.sum(pattern_present) if n_img > 0 else 0
        detection_counts.append(count_detected)
        success_rate = count_detected / n_img if n_img > 0 else 0
        success_rates.append(success_rate)

        time_per_img = elapsed / n_img if n_img > 0 else 0
        time_per_images.append(time_per_img)

        speedup = time_per_image_orig / time_per_img if time_per_img > 0 else 0
        speedups.append(speedup)

        print(f'Average test statistic: {avg_stat:.4f}, Time: {elapsed:.2f}s, '
              f'Detections: {count_detected}/{n_img}, Success rate: {success_rate:.2%}, '
              f'Time per image: {time_per_img:.4f}s, Speedup vs original: {speedup:.2f}x')

    # Add original image stats at the end for comparison
    avg_stats.append(np.mean(statistic_orig))
    runtimes.append(elapsed_orig)
    n_images.append(n_orig)
    detection_counts.append(np.sum(pattern_present_orig))
    success_rates.append(np.sum(pattern_present_orig) / n_orig if n_orig > 0 else 0)
    time_per_images.append(time_per_image_orig)
    speedups.append(1.0)  # Original vs original speedup = 1

    # Save results to a CSV with mixed types handled properly
    with open('crop_vs_original_results.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['CropSize', 'AvgStatistic', 'TotalTime(s)', 'NumImages', 'Detections', 'SuccessRate', 'TimePerImage(s)', 'SpeedupVsOriginal'])
        for i in range(len(crop_labels)):
            writer.writerow([
                crop_labels[i],
                avg_stats[i],
                runtimes[i],
                n_images[i],
                detection_counts[i],
                success_rates[i],
                time_per_images[i],
                speedups[i]
            ])

    # Plotting results
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.plot(crop_labels, avg_stats, marker='o')
    plt.title("Average Test Statistic vs Crop Size")
    plt.xlabel("Crop Size (px)")
    plt.ylabel("Average Test Statistic")
    plt.grid(True)

    plt.subplot(1, 3, 2)
    plt.plot(crop_labels, time_per_images, marker='o', color='orange')
    plt.title("Average Time per Image vs Crop Size")
    plt.xlabel("Crop Size (px)")
    plt.ylabel("Time per Image (s)")
    plt.grid(True)

    plt.subplot(1, 3, 3)
    plt.plot(crop_labels, speedups, marker='o', color='green')
    plt.title("Speedup vs Original Image Processing")
    plt.xlabel("Crop Size (px)")
    plt.ylabel("Speedup Factor")
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("crop_vs_original_analysis.png")
    plt.show()

if __name__ == '__main__':
    main()