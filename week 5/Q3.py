import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

# A - Read the "noisy_img.png" image in gray
img = cv2.imread('noisy_img.png', cv2.IMREAD_GRAYSCALE)

# B - Obtain the type of image noise distribution by separating a suitable band from the image
strip = img[:img.shape[0] // 2, :img.shape[1] // 3]

plt.figure()
plt.subplot(121)
plt.imshow(strip, cmap='gray', vmin=0, vmax=255)
plt.title('Chosen strip of image (Upper left corner)', fontsize=10)
plt.axis('off')

plt.subplot(122)
plt.hist(strip.ravel(), bins=256, range=(0.0, 255), fc='b', ec='k')
plt.title('Histogram of strip', fontsize=20)
plt.show()


# C - Implement the alpha-trimmed mean filter and apply it to the image with a 5x5 window and d=10
def alpha_trimmed_mean_filter(img, window_size, d):
    filtered_img = cv2.copyMakeBorder(img, window_size // 2, window_size // 2, window_size // 2, window_size // 2,
                                      cv2.BORDER_REFLECT)
    for i in range(window_size // 2, img.shape[0] + window_size // 2):
        for j in range(window_size // 2, img.shape[1] + window_size // 2):
            window = filtered_img[i - window_size // 2:i + window_size // 2 + 1,
                     j - window_size // 2:j + window_size // 2 + 1]
            window_flat = np.ravel(window)
            window_flat_sorted = np.sort(window_flat)
            window_flat_trimmed = window_flat_sorted[d // 2:-d // 2]
            filtered_img[i, j] = np.sum(window_flat_trimmed) / len(window_flat_trimmed)
    filtered_img = filtered_img[window_size // 2:-window_size // 2, window_size // 2:-window_size // 2]
    return filtered_img


img_alpha_trimmed = alpha_trimmed_mean_filter(img, 5, 10)

plt.figure()
plt.subplot(121)
plt.imshow(img, cmap='gray', vmin=0, vmax=255)
plt.title('Noisy Image', fontsize=10)
plt.axis('off')

plt.subplot(122)
plt.imshow(img_alpha_trimmed, cmap='gray', vmin=0, vmax=255)
plt.title('After Alpha-Trimmed Mean Filter', fontsize=10)
plt.axis('off')
plt.show()

# D - Read the image degraded_img.png in gray
img_degraded = cv2.imread('degraded_img.png', cv2.IMREAD_GRAYSCALE)


# E - Restore the original image using inverse filtering and H(u,v) function
def restore_image(img, H, cutoff_frequency):
    M, N = img.shape
    H_resized = cv2.resize(H.astype(np.float32), (N, M))
    F = np.fft.fft2(img)
    H_complex = np.fft.fft2(H_resized)

    Gaussian_filter = np.zeros((M, N), dtype=np.float32)
    for i in range(M):
        for j in range(N):
            Gaussian_filter[i][j] = math.exp(-1 * ((i - M / 2) ** 2 + (j - N / 2) ** 2) / (2 * (cutoff_frequency ** 2)))

    F_filtered = F / (H_complex + 0.001)
    F_restored = F_filtered / np.fft.fft2(Gaussian_filter)

    img_restored = np.abs(np.fft.ifft2(F_restored))
    img_restored = img_restored.astype(np.uint8)
    return img_restored, Gaussian_filter


M, N = img_degraded.shape
H = np.zeros((M, N), dtype=np.complex64)

for u in range(M):
    for v in range(N):
        H[u][v] = math.exp(0.0025 * ((u - M / 2) ** 2 + (v - N / 2) ** 2) ** (5 / 6))

cutoff_frequency = 30
img_restored, Gaussian_filter = restore_image(img_degraded, H, cutoff_frequency)

plt.figure()
plt.subplot(131)
plt.imshow(img_degraded, cmap='gray', vmin=0, vmax=255)
plt.title('Degraded Image', fontsize=10)
plt.axis('off')

plt.subplot(132)
plt.imshow(Gaussian_filter, cmap='gray')
plt.title('Gaussian Filter', fontsize=10)
plt.axis('off')

plt.subplot(133)
plt.imshow(img_restored, cmap='gray', vmin=0, vmax=255)
plt.title('Restored Image', fontsize=10)
plt.axis('off')

plt.show()

# Save the output images using OpenCV
cv2.imwrite('strip.png', strip)
cv2.imwrite('filtered_img.png', img_alpha_trimmed)
cv2.imwrite('restored_img.png', img_restored)

# Show the chosen strip of image
cv2.imshow('Chosen Strip of Image', strip)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Show the noisy image and the filtered image
cv2.imshow('Noisy Image', img)
cv2.imshow('After Alpha-Trimmed Mean Filter', img_alpha_trimmed)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Show the degraded image, Gaussian filter, and restored image
cv2.imshow('Degraded Image', img_degraded)
cv2.imshow('Gaussian Filter', Gaussian_filter)
cv2.imshow('Restored Image', img_restored)
cv2.waitKey(0)
cv2.destroyAllWindows()

