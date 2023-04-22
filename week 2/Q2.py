import cv2
import numpy as np
from matplotlib import pyplot as plt

# read image A and B as grayscale
imgA = cv2.imread('HW2_Q4_FingerPrint.tif', cv2.IMREAD_GRAYSCALE)
imgB = cv2.imread('HW2_Q4_Ultrasound-Fetus.tif', cv2.IMREAD_GRAYSCALE)

# define function for smoothed histogram
def smoothed_histogram(image):
    # Calculate the histogram of the input image
    hist, _ = np.histogram(image, 256, [0, 256])
    # Calculate the cumulative distribution function of the histogram
    cdf = np.cumsum(hist)
    # Normalize the cumulative distribution function so that it has a range of [0, 1]
    cdf_normalized = cdf / cdf.max()
    # Create a lookup table
    lookup_table = np.interp(np.arange(0, 256), np.arange(0, 256), cdf_normalized * 255).astype(np.uint8)
    # Apply the lookup table to the input image to smooth its histogram
    smoothed_image = cv2.LUT(image, lookup_table)
    return smoothed_image

# define function for CLAHE filter
def apply_clahe_filter(image, clipLimit=2.0, tileGridSize=(8, 8)):
    # Create a CLAHE object with the specified clip limit and tile size
    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
    # Apply the CLAHE filter to the input image
    output_image = clahe.apply(image)
    return output_image

# apply functions to image A
smoothed_imgA = smoothed_histogram(imgA)
clahe_imgA = apply_clahe_filter(imgA)

# apply functions to image B
smoothed_imgB = smoothed_histogram(imgB)
clahe_imgB = apply_clahe_filter(imgB)

# create histogram plots for image A
plt.figure(figsize=(16, 10))
plt.subplot(3, 2, 1)
plt.imshow(imgA, cmap='gray')
plt.title('Input Image A')
plt.subplot(3, 2, 2)
plt.hist(imgA.ravel(), 256, [0, 256], color='gray')
plt.title('Histogram of Input Image A')
plt.subplot(3, 2, 3)
plt.imshow(smoothed_imgA, cmap='gray')
plt.title('Smoothed Image A')
plt.subplot(3, 2, 4)
plt.hist(smoothed_imgA.ravel(), 256, [0, 256], color='gray')
plt.title('Histogram of Smoothed Image A')
plt.subplot(3, 2, 5)
plt.imshow(clahe_imgA, cmap='gray')
plt.title('CLAHE Filtered Image A')
plt.subplot(3, 2, 6)
plt.hist(clahe_imgA.ravel(), 256, [0, 256], color='gray')
plt.title('Histogram of CLAHE Filtered Image A')
plt.suptitle('Histogram Plots for Image A')
plt.show()

# create cumulative histogram distribution plots for image A
plt.figure(figsize=(16, 10))
plt.subplot(3, 2, 1)
plt.imshow(imgA, cmap='gray')
plt.title('Input Image A')
plt.subplot(3, 2, 2)
plt.hist(imgA.ravel(), 256, [0, 256], color='gray')
plt.title('Histogram of Input Image A')
plt.subplot(3, 2, 3)
plt.plot(np.cumsum(imgA.ravel()), color='gray')
plt.title('Cumulative Histogram Distribution of Input Image A')
plt.subplot(3, 2, 4)
plt.plot(np.cumsum(smoothed_imgA.ravel()), color='gray')
plt.title('Cumulative Histogram Distribution of Smoothed Image A')
plt.subplot(3, 2, 5)
plt.plot(np.cumsum(clahe_imgA.ravel()), color='gray')
plt.title('Cumulative Histogram Distribution of CLAHE Filtered Image A')
plt.suptitle('Cumulative Histogram Distribution Plots for Image A')
plt.show()

# create histogram plots for image B
plt.figure(figsize=(16, 10))
plt.subplot(3, 2, 1)
plt.imshow(imgB, cmap='gray')
plt.title('Input Image B')
plt.subplot(3, 2, 2)
plt.hist(imgB.ravel(), 256, [0, 256], color='gray')
plt.title('Histogram of Input Image B')
plt.subplot(3, 2, 3)
plt.imshow(smoothed_imgB, cmap='gray')
plt.title('Smoothed Image B')
plt.subplot(3, 2, 4)
plt.hist(smoothed_imgB.ravel(), 256, [0, 256], color='gray')
plt.title('Histogram of Smoothed Image B')
plt.subplot(3, 2, 5)
plt.imshow(clahe_imgB, cmap='gray')
plt.title('CLAHE Filtered Image B')
plt.subplot(3, 2, 6)
plt.hist(clahe_imgB.ravel(), 256, [0, 256], color='gray')
plt.title('Histogram of CLAHE Filtered Image B')
plt.suptitle('Histogram Plots for Image B')
plt.show()

# create cumulative histogram distribution
# plots for image B
plt.figure(figsize=(16, 10))
plt.subplot(3, 2, 1)
plt.imshow(imgB, cmap='gray')
plt.title('Input Image B')
plt.subplot(3, 2, 2)
plt.hist(imgB.ravel(), 256, [0, 256], color='gray')
plt.title('Histogram of Input Image B')
plt.subplot(3, 2, 3)
plt.plot(np.cumsum(imgB.ravel()), color='gray')
plt.title('Cumulative Histogram Distribution of Input Image B')
plt.subplot(3, 2, 4)
plt.plot(np.cumsum(smoothed_imgB.ravel()), color='gray')
plt.title('Cumulative Histogram Distribution of Smoothed Image B')
plt.subplot(3, 2, 5)
plt.plot(np.cumsum(clahe_imgB.ravel()), color='gray')
plt.title('Cumulative Histogram Distribution of CLAHE Filtered Image B')
plt.suptitle('Cumulative Histogram Distribution Plots for Image B')
plt.show()

# display images
cv2.imshow('Image A', imgA)
cv2.imshow('Smoothed Image A', smoothed_imgA)
cv2.imshow('CLAHE Filtered Image A', clahe_imgA)
cv2.imshow('Image B', imgB)
cv2.imshow('Smoothed Image B', smoothed_imgB)
cv2.imshow('CLAHE Filtered Image B', clahe_imgB)
cv2.waitKey()
