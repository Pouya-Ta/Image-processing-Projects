import numpy as np
import matplotlib.pyplot as plt

# Define the intensity levels and image sizes
num_levels = 3
image_size = 5

# Define the images
A = np.array([(0, 0, 0, 1, 0),
              (0, 0, 1, 0, 0),
              (1, 1, 1, 1, 1),
              (0, 0, 1, 0, 0),
              (0, 0, 1, 0, 2)])
B = np.array([(0, 0, 0, 0, 1),
              (0, 0, 0, 0, 1),
              (0, 0, 2, 2, 1),
              (0, 0, 2, 2, 1),
              (1, 1, 1, 1, 1)])

# Create an empty 2D histogram with dimensions num_levels x num_levels
histogram = np.zeros((num_levels, num_levels), dtype=np.int)

# Loop through each pixel in both images
for i in range(image_size):
    for j in range(image_size):
        # Get the intensity values of the corresponding pixels in both images
        intensity_a = A[i][j]
        intensity_b = B[i][j]

        # Increment the corresponding bin in the histogram
        histogram[intensity_a, intensity_b] += 1

# Show the histogram using Matplotlib
plt.imshow(histogram, cmap='gray', extent=[0, num_levels-1, 0, num_levels-1],
           origin='lower', interpolation='nearest')
plt.colorbar()
plt.xlabel('Image B')
plt.ylabel('Image A')
plt.show()
