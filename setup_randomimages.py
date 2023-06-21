import os
import random
import shutil

def select_random_images(input_folder, output_folder, num_images=200):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    image_files = [file for file in os.listdir(input_folder) if file.endswith('.jpg')]

    selected_images = random.sample(image_files, min(num_images, len(image_files)))

    for image_file in selected_images:
        src_path = os.path.join(input_folder, image_file)
        dst_path = os.path.join(output_folder, image_file)

        shutil.copy2(src_path, dst_path)

# Example usage
input_folder = "/Users/gustavorubio/Downloads/AFRL Data/13963_1920x1080_02"
output_folder = "/Users/gustavorubio/Downloads/AFRL Data/Fake200"

select_random_images(input_folder, output_folder, num_images=200)

# Use this to select random images from dataset