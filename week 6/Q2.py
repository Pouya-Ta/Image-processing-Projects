import cv2
import matplotlib.pyplot as plt
import numpy as np

# Load image A
imageA = cv2.imread('HW6Q2A.png', cv2.IMREAD_GRAYSCALE)
imageA = np.array(imageA)

# Load image B
imageB = cv2.imread('HW6Q2B.png', cv2.IMREAD_GRAYSCALE)
imageB = np.array(imageB)

# Define the intensity levels
I = [0, 1, 2]  # black: 0-84, gray: 85-169, white: 170-255

# Create a 3D histogram with dimensions 3x3x2
hist = np.zeros((3, 3, 2), dtype=np.int32)

# Loop through each pixel in both images
for i in range(25):
    for j in range(2):
        # Get the color intensity of the pixel in image A and B
        intensity_a = np.digitize(imageA[i][j], [85, 170])
        intensity_b = np.digitize(imageB[i][j], [85, 170])

        # Increment the appropriate bin in the histogram
        hist[intensity_a][intensity_b][j] += 1

# Reshape the histogram into a 2D array for plotting
hist_2d = hist.reshape((9, 2))

# Plot the histogram using Matplotlib
plt.imshow(hist_2d.T, cmap='gray')
plt.xlabel('Image A')
plt.ylabel('Image B')
plt.show()
