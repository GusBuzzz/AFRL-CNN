import os
import cv2
import numpy as np
import matplotlib.pyplot as plts
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers

# Set the CUDA visible devices to use a specific GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Replace '0' with the index of the desired GPU

# Function to load and preprocess images from a folder
def load_images_from_folder(folder, label):
    images = []
    labels = []
    for filename in os.listdir(folder):
        if filename.endswith(".jpg"):
            img = cv2.imread(os.path.join(folder, filename))
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                #img = cv2.resize(img, (1920, 1080))
                images.append(img)
                labels.append(label)
    return images, labels

# Define the paths to your real and synthetic image folders
real_folder = "C:\\Users\\redForce 1\\Downloads\\AFRL\\AFRL Data\\AFRL Data\\Real200"
synthetic_folder = "C:\\Users\\redForce 1\\Downloads\\AFRL\\AFRL Data\\AFRL Data\\Fake200"

# Load and preprocess real images
print("Preprocessing real images and labels")
real_images, real_labels = load_images_from_folder(real_folder, 0)
print("Done")

# Load and preprocess synthetic images
print("Preprocessing synthetic images and labels")
synthetic_images, synthetic_labels = load_images_from_folder(synthetic_folder, 1)
print("Done")

# Combine real and synthetic images and labels
images = real_images + synthetic_images
labels = real_labels + synthetic_labels

# Convert the image and label lists to numpy arrays
images = np.array(images)
labels = np.array(labels)

# Shuffle the data
indices = np.arange(len(images))
np.random.shuffle(indices)
images = images[indices]
labels = labels[indices]

# Split the data into training and testing sets
split_ratio = 0.8
validation_ratio = 0.1
test_ratio = 0.1

# Calculate the number of samples for each set
total_samples = len(images)
num_train_samples = int(total_samples * split_ratio)
num_val_samples = int(total_samples * validation_ratio)
num_test_samples = int(total_samples * test_ratio)

# Split the images and labels
train_images = images[:num_train_samples]
train_labels = labels[:num_train_samples]

val_images = images[num_train_samples:num_train_samples + num_val_samples]
val_labels = labels[num_train_samples:num_train_samples + num_val_samples]

test_images = images[num_train_samples + num_val_samples:]
test_labels = labels[num_train_samples + num_val_samples:]

# Preprocess the image data by scaling it to a range of 0 to 1
train_images = train_images / 255.0
val_images = val_images / 255.0
test_images = test_images / 255.0

# Convert labels to one-hot encoded vectors
train_labels = train_labels.reshape(-1, 1)
val_labels = val_labels.reshape(-1, 1)
test_labels = test_labels.reshape(-1, 1)

# Create the CNN model
print('Creating model')
model = keras.Sequential([
    # Block One
    # Batch Normalization layers help in stabilizing the training process, improves generalization, and accelerates convergence.
    layers.Conv2D(filters=32, kernel_size=3, activation='relu', padding='same', input_shape=[1080, 1920, 3]),
    #layers.BatchNormalization(),
    layers.MaxPool2D(),

    # Block Two
    layers.Conv2D(filters=64, kernel_size=3, activation='relu', padding='same'),
    #layers.BatchNormalization(),
    layers.MaxPool2D(),

    # Block Three
    layers.Conv2D(filters=128, kernel_size=3, activation='relu', padding='same'),
    layers.Conv2D(filters=128, kernel_size=3, activation='relu', padding='same'),
    #layers.BatchNormalization(),
    layers.MaxPool2D(),

    # Head
    # The size of the dense layer has been increased to 256 neurons, allowing for a more expressive and powerful representation
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    #layers.BatchNormalization(),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid'),
])

print('Done')

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
print('Training the model')
model.fit(train_images, train_labels, epochs=1, batch_size=16, validation_data=(val_images, val_labels))
print('Done')

# Save the trained model
model.save("C:\\Users\\redForce 1\\Downloads\\AFRL\\Models\\model2.1_1920x1080.h5")
print("Model saved.")

# Evaluate the model
loss, accuracy = model.evaluate(test_images, test_labels)
print("Test Accuracy:", accuracy)
