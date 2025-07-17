import glob
import cv2
import numpy as np

# Set the path to the directory containing the images
# path = './lof/*.tif'
path = 'lof\\10349.tif'

# Iterate through each image in the directory
for file in glob.glob(path):
    # Open the image in grayscalepytho
    img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)

    # Calculate the maximum intensity
    max_intensity = np.amax(img)

    # Print "gas" if the intensity is greater than or equal to 63, or "lof" otherwise
    if max_intensity >= 63:
        print("gas")
    else:
        print("lof")
