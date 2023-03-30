import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
# first, we have to define libraries which we want to use in the first question.

# section 1
img1 = cv.imread('chest-xray.png')
print("shape (Original Image): ", img1.shape)

imgGrayScale = cv.imread('chest-xray.png', cv.IMREAD_GRAYSCALE)
print("shape (Grayscale Image): ", imgGrayScale.shape)

# section 2
print("d-type (Grayscale Image): ", imgGrayScale.dtype)

# section 3
print("size (Original Image): ", (img1.size * img1.itemsize))
print("size (Grayscale Image): ", (imgGrayScale.size * imgGrayScale.itemsize))

# section 4
fig_first = plt.figure()
left_side = imgGrayScale[100:493, 300:500]
plt.axis('off')
plt.title("left_side grayscale")
plt.imshow(left_side, cmap='gray')

# section 5
fig_2nd = plt.figure()
rows, cols = left_side.shape
ref = np.float32([[1, 0, 0], [0, -1, rows], [0, 0, 1]])
reflected = cv.warpPerspective(left_side, ref, (cols, rows))
plt.axis('off')
plt.title("x-axis reflection")
plt.imshow(reflected, cmap='gray')

# section 6
# V_min = 0
# V_max = 255 as we know

fig, ((f1, f2, f3), (f4, f5, f6)) = plt.subplots(2, 3)
f1.set_title("grayscale image")
f1.imshow(imgGrayScale, cmap='gray', vmin=0, vmax=255)
f2.set_title("left_side image")
f2.imshow(left_side, cmap='gray', vmin=0, vmax=255)
f3.set_title("left_side's x-axis reflection image")
f3.imshow(reflected, cmap='gray', vmin=0, vmax=255)

# section 7
range_7 = range(0, 256, 4)  # as we know it's not going to be counted and the last one is 255
bins = (*range_7, 255)
f4.set_title("grayscale HISTOGRAM")
f4.hist(imgGrayScale.ravel(), bins=bins, rwidth=0.6)

f5.set_title("left_side HISTOGRAM")
f5.hist(left_side.ravel(), bins=bins, rwidth=0.6)

f6.set_title("reflected HISTOGRAM")
f6.hist(reflected.ravel(), bins=bins, rwidth=0.6)

fig.tight_layout(pad=2.0)
# here we increase the size between outputs

plt.show()
