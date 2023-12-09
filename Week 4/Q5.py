import numpy as np
import cv2
import matplotlib.pyplot as plt

# Making an empty image and add things we need
img = np.zeros((200,200), dtype=np.uint8)
rect_width = 80
rect_height = 60
x1 = int(200/2 - rect_width/2)
y1 = int(200/2 - rect_height/2)
x2 = x1 + rect_width
y2 = y1 + rect_height
cv2.rectangle(img, (x1, y1), (x2, y2), 255, -1)

# Make changes to the image
tx = 30
ty = 0
M = np.float32([[1, 0, -tx],[0, 1, -ty]])
img_translated1 = cv2.warpAffine(img, M, (200, 200))
tx = 0
ty = 20
M = np.float32([[1, 0, -tx], [0, 1, -ty]])
img_translated2 = cv2.warpAffine(img_translated1, M, (200, 200))
M = cv2.getRotationMatrix2D((200/2, 200/2), 40, 1)
img_rotated1 = cv2.warpAffine(img_translated2, M, (200, 200))
M = cv2.getRotationMatrix2D((200/2, 200/2), -90, 1)
img_rotated2 = cv2.warpAffine(img_rotated1, M, (200, 200))

# Fourier transform and display images and the logarithm of size and secondary phase
img_list = [img, img_translated1, img_translated2, img_rotated1, img_rotated2]
for i in range(len(img_list)):
    f = np.fft.fft2(img_list[i])
    plt.subplots_adjust(wspace=0.5, hspace=0.5)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20*np.log(np.abs(fshift))
    phase_spectrum = np.angle(fshift)
    plt.subplot(5,3,i*3+1),plt.imshow(img_list[i], cmap = 'gray')
    plt.title('Input Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(5,3,i*3+2),plt.imshow(magnitude_spectrum, cmap = 'gray')
    plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
    plt.subplot(5,3,i*3+3),plt.imshow(phase_spectrum, cmap = 'gray')
    plt.title('Phase Spectrum'), plt.xticks([]), plt.yticks([])
plt.show()

# Draw a graph of the values of two frequencies 0
f = np.fft.fft2(img)
fshift = np.fft.fftshift(f)
rows, cols = img.shape
zero_freq_row = int(rows/2)
zero_freq_col = int(cols/2)
fshift[zero_freq_row, zero_freq_col+1] *= 2
fshift[zero_freq_row+1, zero_freq_col] *= 2
magnitude_spectrum = np.abs(fshift)
x = np.arange(0, rows)
y = np.arange(0, cols)
X, Y = np.meshgrid(x, y)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(X, Y, magnitude_spectrum, cmap='viridis')
plt.title('2D Plot of Zero Frequencies')
plt.xlabel('Rows'), plt.ylabel('Columns')
plt.show()
