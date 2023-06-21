import os
import shutil

def filter_images(folder_path, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(folder_path):
        if filename.endswith(".jpg") and "rgb_0000" in filename:
            src_path = os.path.join(folder_path, filename)
            dst_path = os.path.join(output_folder, filename)
            shutil.copy2(src_path, dst_path)

folder_path = "/Volumes/LaCie/synthetic/New Synthetic Data/13963_1920x1080_02"
output_folder = "/Volumes/LaCie/synthetic/New Synthetic Data/13963_1920x1080_02 filtered"

filter_images(folder_path, output_folder)
