import os
import random
import shutil

def select_random_images(main_folder, output_folder, num_images=2):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    subfolders = [folder for folder in os.listdir(main_folder) if os.path.isdir(os.path.join(main_folder, folder))]

    for subfolder_name in subfolders:
        subfolder_path = os.path.join(main_folder, subfolder_name)
        image_files = [file for file in os.listdir(subfolder_path) if file.endswith('.jpg')]

        selected_images = random.sample(image_files, min(num_images, len(image_files)))

        for image_file in selected_images:
            src_path = os.path.join(subfolder_path, image_file)
            dst_path = os.path.join(output_folder, image_file)

            shutil.copy2(src_path, dst_path)

# Example usage
main_folder = "/Volumes/LaCie/real/train_images_folder3"
output_folder = "/Volumes/LaCie/real/real_image_filtered"

select_random_images(main_folder, output_folder, num_images=2)
