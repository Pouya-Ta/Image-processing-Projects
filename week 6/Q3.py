import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np



# Read image A in gray
img = cv.imread('normal-ct-of-the-neck.jpg', cv.IMREAD_GRAYSCALE)

# Apply Gaussian filter to smooth the image
img_blur = cv.GaussianBlur(img, (5, 5), 0)

# Calculate gradients using Sobel and Prewitt kernels
sobel1 = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
sobel2 = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
sobel3 = np.array([[0, 1, 2], [-1, 0, 1], [-2, -1, 0]])
sobel4 = np.array([[-2, -1, 0], [-1, 0, 1], [0, 1, 2]])



S_result1 = cv.filter2D(img_blur, -1, sobel1)
S_result2 = cv.filter2D(img_blur, -1, sobel2)
S_result3 = cv.filter2D(img_blur, -1, sobel3)
S_result4 = cv.filter2D(img_blur, -1, sobel4)

sobel_results = S_result1 + S_result2 + S_result3 + S_result4

Prewitt1 = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
Prewitt2 = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
Prewitt3 = np.array([[0, 1, 1], [-1, 0, 1], [-1, -1, 0]])
Prewitt4 = np.array([[-1, -1, 0], [-1, 0, 1], [0, 1, 1]])

P_result1 = cv.filter2D(img_blur, -1, Prewitt1)
P_result2 = cv.filter2D(img_blur, -1, Prewitt2)
P_result3 = cv.filter2D(img_blur, -1, Prewitt3)
P_result4 = cv.filter2D(img_blur, -1, Prewitt4)

prewitt_results = P_result1 + P_result2 + P_result3 + P_result4


# Display results of Sobel and Prewitt filters
plt.figure()
plt.suptitle('Sobel Kernels', fontsize=20)
plt.subplot(141), plt.imshow(S_result1, cmap='gray', vmin=0, vmax=255), plt.title('Vertical edges', fontsize=10), plt.axis('off')
plt.subplot(142), plt.imshow(S_result2, cmap='gray', vmin=0, vmax=255), plt.title('Horizontal edges', fontsize=10), plt.axis('off')
plt.subplot(143), plt.imshow(S_result3, cmap='gray', vmin=0, vmax=255), plt.title('Diagonal edges', fontsize=10), plt.axis('off')
plt.subplot(144), plt.imshow(S_result4, cmap='gray', vmin=0, vmax=255), plt.title('Diagonal edges', fontsize=10), plt.axis('off')

plt.figure()


plt.suptitle('Prewitt Kernels', fontsize =20)
plt.subplot(141), plt.imshow(P_result1, cmap='gray', vmin=0, vmax=255), plt.title('Vertical edges', fontsize=10), plt.axis('off')
plt.subplot(142), plt.imshow(P_result2, cmap='gray', vmin=0, vmax=255), plt.title('Horizontal edges', fontsize=10), plt.axis('off')
plt.subplot(143), plt.imshow(P_result3, cmap='gray', vmin=0, vmax=255), plt.title('Diagonal edges', fontsize=10), plt.axis('off')
plt.subplot(144), plt.imshow(P_result4, cmap='gray', vmin=0, vmax=255), plt.title('Diagonal edges', fontsize=10), plt.axis('off')

plt.figure()

# Display results of Sobel and Prewitt combined filters
plt.suptitle('Combined Kernels (Sobel + Prewitt)', fontsize=20)
plt.subplot(121), plt.imshow(sobel_results, cmap='gray', vmin=0, vmax=255), plt.title('Sobel', fontsize=15), plt.axis('off')
plt.subplot(122), plt.imshow(prewitt_results, cmap='gray', vmin=0, vmax=255), plt.title('Prewitt', fontsize=15), plt.axis('off')

plt.show()