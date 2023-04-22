import cv2
import numpy as np

# Load the image
img = cv2.imread('C:/Users/ASUS/Documents/university/Term6/Image processing/HW/HW3/salt-and-pepper-Skeleton.PNG')

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply median filtering with kernel size 3x3
denoised = cv2.medianBlur(gray, 3)

# Apply a high-pass filter to enhance vertical features (bones)
kernel_v = np.array([[-1,-1,-1], [2,2,2], [-1,-1,-1]])
filtered_v = cv2.filter2D(denoised, -1, kernel_v)

# Apply a high-pass filter to enhance horizontal features (bones)
kernel_h = np.array([[-1,2,-1], [-1,2,-1], [-1,2,-1]])
filtered_h = cv2.filter2D(denoised, -1, kernel_h)

# Invert the colors of the denoised image
inverted = cv2.bitwise_not(denoised)

# Apply a low-pass filter to smooth out noise and highlight muscle regions
kernel_m = np.ones((5,5), np.float32) / 25
filtered_m = cv2.filter2D(inverted, -1, kernel_m)

# Invert the colors of the filtered muscle image back to the original orientation
muscles = cv2.bitwise_not(filtered_m)

# Display the original, denoised, and filtered images
cv2.imshow('Original', gray)
cv2.imshow('Denoised', denoised)
cv2.imshow('Filtered (vertical)', filtered_v)
cv2.imshow('Filtered (horizontal)', filtered_h)
cv2.imshow('Muscles', muscles)
cv2.waitKey(0)
cv2.destroyAllWindows()
