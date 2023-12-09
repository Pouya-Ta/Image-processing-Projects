import cv2 as cv
import matplotlib.pyplot as plt


# Read the image
img = cv.imread('limbal-dermoid.jpeg', cv.IMREAD_GRAYSCALE)

# Apply Gaussian blur to reduce noise
img_blur = cv.GaussianBlur(img, (5, 5), 0)

# Apply Canny edge detection
edges = cv.Canny(img_blur, 100, 200)  # Adjust the threshold values as needed

# Display the original image and the detected edges
plt.subplot(121), plt.imshow(img, cmap='gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(edges, cmap='gray')
plt.title('Canny Edges'), plt.xticks([]), plt.yticks([])

plt.show()