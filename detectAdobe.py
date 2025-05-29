# -*- coding: UTF-8 -*-
"""
@author: Jan Butora (jan.butora@univ-lille.fr)
@author: Patrick Bas (patrick.bas@univ-lille.fr)
CNRS, University of Lille 2024
"""

import numpy as np
from PIL import Image
import prnu
from skimage.util import view_as_blocks
from scipy.stats import norm

def qfunc(x):
  """
  Q function
  :param x: value(s) to compute the Q function of
  :return: the Q function of input value x
  """
  return 0.5 - 0.5*special.erf(x/np.sqrt(2))

def inv_qfunc(x):
  """
  Inverse Q function
  :param x: value(s) to compute the Inverse Q function of
  :return: the Inverse Q function of input value x
  """
  return norm.ppf(1-x)

def get_threshold(alpha, mu, sigma):
  """
  Finds a threshold of a desired FPR alpha for a Gaussian distribution with mean mu and std sigma. The values above this threshold will be considered as containing the Adobe pattern
  :param alpha: Desired False Positive Rate
  :param mu: mean of the Gaussian distribution
  :param sigma: standard deviation of the Gaussian distribution
  :return: The decision threshold
  """
  return mu + sigma*inv_qfunc(alpha)


def test_statistic_roll(im_name, watermark):
    """
    Computes test statistics for all possible shifts of a tiled watermark pattern.
    Assumes input image is <= 512x512 pixels (no rotation handling).
    This was written by Author Charlie Cameron @ 29th May 2025
    Args:
        im_name: Path to input image (max 512x512)
        watermark: Adobe pattern (h_w, w_w)
        
    Returns:
        dict: {
            'best_T': maximum test statistic,
            'best_shift': (dy, dx) for best_T,
            'all_ts': array of test statistics for all shifts,
            'all_shifts': list of (dy, dx) tuples
        }
    """
    # Load and validate image size
    im = np.float32(np.array(Image.open(im_name)))
    assert max(im.shape) <= 512, "Image must be <= 512x512 pixels"
    
    h_w, w_w = watermark.shape
    res = prnu.extract_single(im) if im.ndim == 3 else im
    res = res[:res.shape[0]//h_w*h_w, :res.shape[1]//w_w*w_w]
    
    # Precompute normalized watermark
    W_norm = (watermark - watermark.mean()) / np.std(watermark)
    all_ts = np.zeros((h_w, w_w))
    
    # Compute statistics for all shifts
    for dy in range(h_w):
        for dx in range(w_w):
            shifted_res = np.roll(res, shift=(-dy, -dx), axis=(0, 1))
            blocks = view_as_blocks(shifted_res, (h_w, w_w))
            blocks_norm = (blocks - np.mean(blocks, axis=(-1,-2), keepdims=True)) / np.std(blocks, axis=(-1,-2), keepdims=True)
            all_ts[dy, dx] = np.mean(W_norm * blocks_norm) * np.sqrt(blocks.size)
    
    # Find best shift
    best_idx = np.unravel_index(np.argmax(all_ts), all_ts.shape)
    return {
        'best_T': all_ts[best_idx],
        'best_shift': best_idx,
        'all_ts': all_ts,
        'all_shifts': [(dy, dx) for dy in range(h_w) for dx in range(w_w)]
    }

def test_statistic(im_name,watermark):
  """
  Computes the test statistic as given by Equation (15)
  :param im_name: Image path of an image
  :param watermark: array containing the Adobe pattern
  :return: The test statistic (15)
  """
  im = np.float32(np.array(Image.open(im_name)))
  h_w, w_w = watermark.shape

  h,w = im.shape[:2]

  # If image is in portrait orientation, rotate
  if h> w:
      im = np.rot90(im, 1)
      h,w = w,h

  res = prnu.extract_single(im)

  res = res[:res.shape[0]//h_w*h_w,:res.shape[1]//w_w*w_w]
  res_view = view_as_blocks(res, (h_w,w_w))
  corr = np.mean((watermark-watermark.mean())*(res_view-np.mean(res_view, axis=(-1,-2)).reshape(res_view.shape[0], res_view.shape[1],1,1)), axis=(-1,-2))/np.std(watermark)/np.std(res_view,axis=(-1,-2))
  T = np.mean(corr)*np.sqrt(corr.size)

  return T

def detect_adobe_pattern(im_name, adobe_pattern, FPR=1e-5):
  """
  Computes if a given image contains the adobe pattern
  :param im_name: Image path of an image
  :param adobe_pattern: array containing the Adobe pattern
  :param FPR: Desired False Positive Rate
  :return: True if the Adobe pattern has been detected in the given image with a prescribed FPR, False otherwise. Additionally returns the test statistic T, as given by equation (15)
  """
  T = test_statistic(im_name, adobe_pattern)
  mu = 0
  sigma = 1/np.sqrt(adobe_pattern.size)
  threshold = get_threshold(FPR, mu, sigma)
  return T>threshold, T
