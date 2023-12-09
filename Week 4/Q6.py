import cv2
import numpy as np


def butterworth_lowpass_filter(img, radius, n):
    rows, cols = img.shape
    crow, ccol = rows // 2, cols // 2

    # create a mask first, center square is 1, remaining all zeros
    mask = np.zeros((rows, cols), np.uint8)
    for i in range(rows):
        for j in range(cols):
            if np.sqrt((i - crow) ** 2 + (j - ccol) ** 2) < radius:
                mask[i, j] = 1

    # apply the mask on Fourier Transform of the image
    f_shift = np.fft.fftshift(np.fft.fft2(img))
    f_shift_filtered = f_shift * mask
    f_filtered = np.fft.ifft2(np.fft.ifftshift(f_shift_filtered))

    # normalize the output image
    filtered = np.abs(f_filtered)
    filtered = (filtered - np.min(filtered)) / (np.max(filtered) - np.min(filtered))

    return filtered


def butterworth_highpass_filter(img, radius, n):
    rows, cols = img.shape
    crow, ccol = rows // 2, cols // 2

    # create a mask first, center square is 0, remaining all ones
    mask = np.ones((rows, cols), np.uint8)
    for i in range(rows):
        for j in range(cols):
            if np.sqrt((i - crow) ** 2 + (j - ccol) ** 2) < radius:
                mask[i, j] = 0

    # apply the mask on Fourier Transform of the image
    f_shift = np.fft.fftshift(np.fft.fft2(img))
    f_shift_filtered = f_shift * mask
    f_filtered = np.fft.ifft2(np.fft.ifftshift(f_shift_filtered))

    # normalize the output image
    filtered = np.abs(f_filtered)
    filtered = (filtered - np.min(filtered)) / (np.max(filtered) - np.min(filtered))

    return filtered


def gaussian_lowpass_filter(img, radius):
    rows, cols = img.shape
    crow, ccol = rows // 2, cols // 2

    # create a low-pass Gaussian filter
    mask = np.zeros((rows, cols), dtype=np.float32)
    for i in range(rows):
        for j in range(cols):
            d = np.sqrt((i - crow) ** 2 + (j - ccol) ** 2)
            mask[i, j] = np.exp(-d ** 2 / (2 * radius ** 2))

    # apply the mask on Fourier Transform of the image
    f_shift = np.fft.fftshift(np.fft.fft2(img))
    f_shift_filtered = f_shift * mask
    f_filtered = np.fft.ifft2(np.fft.ifftshift(f_shift_filtered))

    # normalize the output image
    filtered = np.abs(f_filtered)
    filtered = (filtered - np.min(filtered)) / (np.max(filtered) - np.min(filtered))

    return filtered


def filter_image(image, filter_type, *args):
    if filter_type == 'butterworth_lowpass':
        radius, n = args
        img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return butterworth_lowpass_filter(img_gray, radius, n)
    elif filter_type == 'butterworth_highpass':
        radius, n = args
        img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return butterworth_highpass_filter(img_gray, radius, n)
    elif filter_type == 'gaussian_lowpass':
        radius = args[0]
        img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return gaussian_lowpass_filter(img_gray, radius)


# read the shoulder image
img = cv2.imread('shoulder.jpg')

# apply filters with different radii
filtered1 = filter_image(img, 'butterworth_lowpass', 50, 2)
filtered2 = filter_image(img, 'butterworth_highpass', 100, 2)
filtered3 = filter_image(img, 'gaussian_lowpass', 200)

# display the results
cv2.imshow('original', img)
cv2.imshow('butterworth_lowpass (radius=50)', filtered1)
cv2.imshow('butterworth_highpass (radius=100)', filtered2)
cv2.imshow('gaussian_lowpass (radius=200)', filtered3)
cv2.waitKey(0)
cv2.destroyAllWindows()
