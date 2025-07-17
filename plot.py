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
# Create an array to store the maximum intensity of each image in the 'gas' folder
max_intensities = []
indexes = []

# Iterate through each image in the 'gas' folder
for i in range(len(gas_images)):
    # Print the current image
    gas_image = gas_images[i]
    fig, ax = plt.subplots()
    # ax.imshow(gas_image, cmap='gray')

    # def on_mouse_move(event):
    #     if event.inaxes:
    #         x, y = event.xdata, event.ydata
    #         intensity_value = gas_image[int(y), int(x)]
    #         print(f'Intensity value at ({x}, {y}): {intensity_value}')

    # cid = fig.canvas.mpl_connect('motion_notify_event', on_mouse_move)
    # plt.show()

    # Calculate the maximum intensity of the current image
    max_intensity = np.amax(gas_image)
    max_intensity_index = np.argmax(gas_image)
    max_intensity_position = np.unravel_index(
        max_intensity_index, gas_image.shape)
    print(
        f'Position of the pixel with the highest intensity: {max_intensity_position}')
    print(f'Maximum intensity: {max_intensity}')

    # Add the maximum intensity to the array
    max_intensities.append(max_intensity)
    indexes.append(1)
# Print the array of maximum intensities
print(
    f'Maximum intensities of all images in the \'gas\' folder: {max_intensities}')


# Iterate through each image in the 'gas' folder
for i in range(len(lof_images)):
    # Print the current image
    gas_image = lof_images[i]
    fig, ax = plt.subplots()
    # ax.imshow(gas_image, cmap='gray')

    # def on_mouse_move(event):
    #     if event.inaxes:
    #         x, y = event.xdata, event.ydata
    #         intensity_value = gas_image[int(y), int(x)]
    #         print(f'Intensity value at ({x}, {y}): {intensity_value}')

    # cid = fig.canvas.mpl_connect('motion_notify_event', on_mouse_move)
    # plt.show()

    # Calculate the maximum intensity of the current image
    max_intensity = np.amax(gas_image)
    max_intensity_index = np.argmax(gas_image)
    max_intensity_position = np.unravel_index(
        max_intensity_index, gas_image.shape)
    print(
        f'Position of the pixel with the highest intensity: {max_intensity_position}')
    print(f'Maximum intensity: {max_intensity}')

    # Add the maximum intensity to the array
    max_intensities.append(max_intensity)
    indexes.append(0)


# Print the array of maximum intensities
print(
    f'Maximum intensities of all images in the \'gas\' folder: {max_intensities}')

pred = []
for i in max_intensities:
    if i >= 63:
        pred.append(1)
    else:
        pred.append(0)

num_correct = 0
for i in range(len(pred)):
    if pred[i] == indexes[i]:
        num_correct += 1
# Calculate the accuracy
accuracy = num_correct / len(pred)

print(f'Accuracy of the classification: {accuracy}')
