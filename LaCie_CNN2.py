'''
AFRL Internship Summer 2023
Author: Gustavo Rubio
Date: 06/21/2023
Purpose: This file trains a model to classify images as either real or synthetic. The model is designed to handle input images of size 1920x1080 and make 
accurate predictions. Once the training process, executed by the 'fix()' method, is completed, the model will be saved in the LaCieModelSave folder.
'''
import os
import cv2
import numpy as np
import matplotlib.pyplot as plts
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
from keras.utils import to_categorical

# Function to load and preprocess images from a folder
def load_images_from_folder(folder, label):
    images = []
    labels = []
    image_paths = []  # Add an empty list to store the image paths
    for filename in os.listdir(folder):
        if filename.endswith(".jpg"):                       # or filename.endswith(".png"):
            img = cv2.imread(os.path.join(folder, filename))
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
                img = cv2.resize(img, (1920, 1080))         # Resize to 1920x1080
                images.append(img)
                labels.append(label)
                image_paths.append(os.path.join(folder, filename))  # Store the image path
    return images, labels, image_paths

# Define the paths to your real and synthetic image folders
real_folder = "/Users/gustavorubio/Downloads/AFRL Data/Real200"
synthetic_folder = "/Users/gustavorubio/Downloads/AFRL Data/Fake200"

# Load and preprocess real images
print("preprocessing real images and labels")
real_images, real_labels, real_image_paths = load_images_from_folder(real_folder, 0)
print("Done")

# Load and preprocess synthetic images
print("preprocessing synthetic images and labels")
synthetic_images, synthetic_labels, synthetic_image_paths = load_images_from_folder(synthetic_folder, 1)
print("Done")

# Combine real and synthetic images and labels
images = real_images + synthetic_images
labels = real_labels + synthetic_labels
paths = real_image_paths + synthetic_image_paths

# Convert the image and label lists to numpy arrays
images = np.array(images)
labels = np.array(labels)
paths = np.array(paths)             # image paths are used to determine which image is the model using

# Shuffle the data
indices = np.arange(len(images))
np.random.shuffle(indices)
images = images[indices]
labels = labels[indices]
paths = paths[indices]

# Split the data into training and testing sets
split_ratio = 0.8           # 80%
validation_ratio = 0.1      # 10%
test_ratio = 0.1            # 10%

# Calculate the number of samples for each set
total_samples = len(images)
num_train_samples = int(total_samples * split_ratio)    # calculates for 80% of data being used
num_val_samples = int(total_samples * validation_ratio) # calculates for 10% of data being used
num_test_samples = int(total_samples * test_ratio)      # calculates for 10% of data being used

# Split the images and labels
train_images = images[:num_train_samples]
train_labels = labels[:num_train_samples]

val_images = images[num_train_samples:num_train_samples + num_val_samples]
val_labels = labels[num_train_samples:num_train_samples + num_val_samples]

test_images = images[num_train_samples + num_val_samples:]
test_labels = labels[num_train_samples + num_val_samples:]

test_paths = paths[num_train_samples + num_val_samples:]   # The split index must match with test_images so that the paths match with the image

# Preprocess the image data by scaling it to a range of 0 to 1
train_images = train_images / 255.0
val_images = val_images / 255.0
test_images = test_images / 255.0

# Convert labels to one-hot encoded vectors
train_labels = to_categorical(train_labels)
val_labels = to_categorical(val_labels)
test_labels = to_categorical(test_labels)

# Create the CNN model
print('Creating model')
model = keras.Sequential([
    # Block One
    layers.Conv2D(filters=32, kernel_size=3, activation='relu', padding='same',
                  input_shape=[128, 128, 3]),
    layers.MaxPool2D(),

    # Block Two
    layers.Conv2D(filters=64, kernel_size=3, activation='relu', padding='same'),
    layers.MaxPool2D(),

    # Block Three
    layers.Conv2D(filters=128, kernel_size=3, activation='relu', padding='same'),
    layers.Conv2D(filters=128, kernel_size=3, activation='relu', padding='same'),
    layers.MaxPool2D(),

    # Head
    layers.Flatten(),
    layers.Dense(6, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(1, activation='sigmoid'),
])
print('Done')
# Compile the model
model.compile(
    optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=0.0001),
    loss='binary_crossentropy',
    metrics=['binary_accuracy'],
)

# Train the model
print('Training the model')
model.fit(train_images, train_labels, epochs=10, batch_size=16, validation_data=(val_images, val_labels))
print('Done')
# Save the trained model
model.save('/Users/gustavorubio/Downloads/Python/AFRL CNN/LaCieModelSave/model_1920x1080.h5')
print("Model saved.")
# Change the 'my_model.h5' to something different so that you do not override your models

# Evaluate the model
model.evaluate(test_images, test_labels)