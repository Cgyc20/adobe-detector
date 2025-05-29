import os
from PIL import Image

def top_left_fixed_crop(old_filepath, crop_size=128):
    """
    Crop images from the top-left corner by a fixed size (e.g., 128x128 pixels).
    Saves to a folder ending with the crop size.
    """
    new_folder_name = f"cropped_images_top_left_{crop_size}"
    new_filepath = os.path.join(old_filepath, new_folder_name)

    if not os.path.exists(new_filepath):
        os.makedirs(new_filepath)
        print(f"Created new folder at: {new_filepath}")
    else:
        print(f"{new_filepath} already exists.")

    for filename in os.listdir(old_filepath):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif')):
            img_path = os.path.join(old_filepath, filename)
            try:
                with Image.open(img_path) as img:
                    width, height = img.size

                    # Ensure the crop size does not exceed the image dimensions
                    crop_width = min(crop_size, width)
                    crop_height = min(crop_size, height)

                    # Crop from top-left corner
                    left = 0
                    top = 0
                    right = left + crop_width
                    bottom = top + crop_height

                    cropped_img = img.crop((left, top, right, bottom))

                    new_img_path = os.path.join(new_filepath, filename)
                    cropped_img.save(new_img_path)
                    print(f"Top-left cropped and saved: {filename} to size {crop_width}x{crop_height}")
            except Exception as e:
                print(f"Error processing {filename}: {e}")

if __name__ == "__main__":
    old_filepath = '/Users/charliecameron/CodingHub/CGC_Solutions/AdobeProject/adobe-detector/Data/NonAdobeImages'
    crop_size = 512 # Change this to 256 or any other size if needed

    top_left_fixed_crop(old_filepath, crop_size)