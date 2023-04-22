import cv2
import numpy as np
from matplotlib import pyplot as plt


# Define the power transformation function
def power_transform(img, gamma=1.0):
    # Normalize the pixel intensity range to [0, 1]
    img_norm = (img.astype(np.float32) / 255.0)

    # Apply the power transformation
    out_img = np.power(img_norm, gamma)

    # Rescale the pixel intensity range back to [0, 255] and convert to uint8
    out_img = (out_img * 255).astype(np.uint8)

    return out_img


# Define the contrast stretching function
def contrast_stretching(img):
    # Define the minimum and maximum intensity values for the new range
    a = 0
    b = 2 ** img.itemsize - 1  # Max value for the bit depth of the image

    # Calculate the minimum and maximum intensity values of the original image
    c = np.min(img)
    d = np.max(img)

    # Apply the contrast stretching formula to each pixel in the image
    out_img = ((img - c) / (d - c)) * (b - a) + a

    # Save the output image as a variable
    out_var = out_img.copy()

    # Return the output image
    return out_var


# Define power law
def power_law(img, y):
    # Define tbe maximum intensity value for the bit depth of the image
    b = 2 ** img.itemsize - 1

    # Normalize pixel intensity to [0, 1]
    img_norm = (img.astype(np.float32) / b)

    # Calculate power law
    out_img = b * np.power(img_norm, y)

    # Save output image
    out_var = out_img.copy()

    # Return output image
    return out_var


# Read the input image as grayscale
img = cv2.imread('HW2_Q3_spine.tif', cv2.IMREAD_GRAYSCALE)

# Define an appropriate value of y for power transformation
y = 0.7

# Apply contrast stretching transformation to the input image
out_contrast = contrast_stretching(img)

# Apply power transformation to the input image with value of y
out_power = power_law(img, y)

# Display the input image and the two transformed images side-by-side
fig, axs = plt.subplots(1, 3, figsize=(10, 5))
axs[0].imshow(img, cmap='gray')
axs[0].set_title('Original Image')

axs[1].imshow(out_contrast, cmap='gray')
axs[1].set_title('Contrast Stretching')

axs[2].imshow(out_power, cmap='gray')
axs[2].set_title('Power Transformation, y=' + str(y))

# Show the final output
plt.tight_layout()
plt.show()
