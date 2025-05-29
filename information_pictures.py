import numpy as np
from glob import glob
import os
from PIL import Image, ExifTags

filepath = '/Users/charliecameron/CodingHub/CGC_Solutions/AdobeProject/adobe-detector/Data/SusClusters/Cluster1/cropped_images/*.jpg'
im_list = np.array(sorted(glob(filepath)))

def print_image_info(image_path):
    print(f"\n--- {image_path} ---")

    try:
        with Image.open(image_path) as img:
            # Basic info
            print(f"Filename       : {os.path.basename(image_path)}")
            print(f"Format         : {img.format}")
            print(f"Mode           : {img.mode}")
            print(f"Size (WxH)     : {img.size}")
            print(f"Info dict      : {img.info}")
            print(f"Is Animated    : {getattr(img, 'is_animated', False)}")
            print(f"Frames         : {getattr(img, 'n_frames', 1)}")

            # Convert to numpy array for analysis
            img_array = np.array(img)

            print(f"Shape          : {img_array.shape}")
            print(f"Dtype          : {img_array.dtype}")
            print(f"Min Pixel Value: {img_array.min()}")
            print(f"Max Pixel Value: {img_array.max()}")
            print(f"Mean Pixel Val : {img_array.mean():.2f}")
            print(f"Std Dev        : {img_array.std():.2f}")

            # EXIF data (if any)
            try:
                exif_data = img._getexif()
                if exif_data:
                    print("EXIF Data:")
                    for tag, value in exif_data.items():
                        tag_name = ExifTags.TAGS.get(tag, tag)
                        print(f"  {tag_name}: {value}")
                else:
                    print("No EXIF data found.")
            except Exception as e:
                print("EXIF extraction failed:", e)

    except Exception as e:
        print(f"Error reading {image_path}: {e}")

print(f"Found {len(im_list)} images.")
for image_path in im_list:
    print_image_info(image_path)