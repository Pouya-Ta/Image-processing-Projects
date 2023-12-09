import cv2
import numpy as np
import matplotlib.pyplot as plt

# read image
img = cv2.imread('chest.tif', 0)

# Apply the Fourier transform
f = np.fft.fft2(img)
fshift = np.fft.fftshift(f)
magnitude_spectrum = 20*np.log(np.abs(fshift))

# Applying the ideal pass filter in the frequency space
rows, cols = img.shape
crow, ccol = rows//2, cols//2
mask = np.zeros((rows, cols), np.uint8)
r = 60
mask[crow-r:crow+r, ccol-r:ccol+r] = 1

# Filter application in frequency space
fshift_filtered = fshift * mask

# Returning the image to space time and reducing distortion
f_ishift = np.fft.ifftshift(fshift_filtered)
img_filtered = 20*np.log(np.abs(np.fft.ifft2(f_ishift)))

# Apply mirroring
dft_shift = np.fft.fftshift(f).astype(np.float32)
mag, phase = cv2.cartToPolar(dft_shift[:, :], np.zeros_like(dft_shift[:, :]))
phase = -phase
[dft_shift[:, :], _] = cv2.polarToCart(mag, phase)
dft_ishift_m = np.fft.ifftshift(dft_shift)

# Returning the image to space time and reducing distortion
img_mir_complex = cv2.idft(dft_ishift_m)
img_mir = cv2.magnitude(img_mir_complex[:, :], np.zeros_like(img_mir_complex[:, :]))

# Image normalization
img_mir = cv2.normalize(img_mir, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

plt.subplot(221), plt.imshow(img, cmap='gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])

plt.subplot(222), plt.imshow(magnitude_spectrum, cmap='gray')
plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])

plt.subplot(223), plt.imshow(img_filtered, cmap='gray')
plt.title('Filtered Image'), plt.xticks([]), plt.yticks([])

plt.subplot(224), plt.imshow(img_mir, cmap='gray')
plt.title('Mirrored Image'), plt.xticks([]), plt.yticks([]) # تغییرات این خط

plt.show()
