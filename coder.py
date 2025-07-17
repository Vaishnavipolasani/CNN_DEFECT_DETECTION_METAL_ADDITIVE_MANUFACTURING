import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt


def load_data(label_type):
    path = f'./{label_type.lower()}/*.tif'
    labels = []
    images = []
    for file in glob.glob(path):
        img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            images.append(img)
            if label_type.lower() == "lof":
                labels.append(0)
            elif label_type.lower() == "gas":
                labels.append(1)
    return np.array(images), np.array(labels)


lof_images, lof_labels = load_data('lof')
gas_images, gas_labels = load_data('gas')

if len(lof_images) == 0 or len(gas_images) == 0:
    print("No images were loaded. Please check the file paths and ensure there are images in the specified directories.")
    exit()

# Create empty arrays to store the predicted and actual labels
predicted_labels = []
actual_labels = []

# Iterate through each image in the 'gas' and 'lof' folders
for i in range(len(gas_images)):
    # Print the current image from the 'gas' folder
    gas_image = gas_images[i]

    # Classify the current image from the 'gas' folder based on the maximum intensity
    max_intensity = np.amax(gas_image)
    if max_intensity > 60:
        predicted_labels.append(1)
    else:
        predicted_labels.append(0)

    # Add the actual label of the current image from the 'gas' folder to the array
    actual_labels.append(gas_labels[i])

for i in range(len(lof_images)):
    # Print the current image from the 'lof' folder
    lof_image = lof_images[i]

    # Classify the current image from the 'lof' folder based on the maximum intensity
    max_intensity = np.amax(lof_image)
    if max_intensity < 60:
        predicted_labels.append(1)
    else:
        predicted_labels.append(0)

    # Add the actual label of the current image from the 'lof' folder to the array
    actual_labels.append(lof_labels[i])

# Calculate the accuracy of the classification
accuracy = np.mean(predicted_labels == actual_labels)
print(f'Accuracy of the classification: {accuracy}')
