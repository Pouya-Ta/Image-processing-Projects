import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def remove(img):
    cv.imshow("image", img)

    def click(event, x, y, flags, param):
        global img
        _, thresh = cv.threshold(img, 20, 255, cv.THRESH_BINARY)
        img_n = thresh
        img_r = np.zeros(img.shape, dtype='uint8')
        test = np.zeros(img.shape, dtype='uint8')
        kernel = cv.getStructuringElement(cv.MORPH_CROSS, (5, 5))
        if event == cv.EVENT_LBUTTONUP:
            print(x, ' ', y)
            img_r[y, x] = 255
            while np.any(img_r != test):
                test = img_r
                img_r = cv.dilate(img_r, kernel, iterations=1)
                img_r = cv.bitwise_and(img_r, img_n)
                cv.imshow("result", img)

            img1 = cv.bitwise_xor(img_r, img)
            img = cv.bitwise_and(img1, img)
            cv.imshow("result", img)
            cv.imwrite('m.png', img)

    while True:
        cv.setMouseCallback('image', click)
        k = cv.waitKey(0)
        if k == 27:
            cv.destroyAllWindows()
            break

    return img


img = cv.imread("circles.png", 0)
remove(img)
result = cv.imread("m.png", 0)


def func(r):
    out = np.zeros((r*2+1, r*2+1), dtype=np.uint8)
    for i in range((r*2)+1):
        for j in range((r*2)+1):
            if np.sqrt((r-i)**2 + (r-j)**2) <= r:
                out[i][j] = 255
            else:
                out[i][j] = 0
    return out


kernel_blob = func(30)
kernel_blob_open = func(60)
kernel_blob_erosion = func(3)
close_blob = cv.morphologyEx(result, cv.MORPH_CLOSE, np.uint8(kernel_blob))
openinng_blob = cv.morphologyEx(close_blob, cv.MORPH_OPEN, np.uint8(kernel_blob_open))
erosion_blob = cv.erode(openinng_blob, np.uint8(kernel_blob_erosion))
edge = openinng_blob - erosion_blob

plt.figure()
plt.subplot(2, 2, 1)
plt.imshow(result, cmap='gray')
plt.subplot(2, 2, 2)
plt.imshow(close_blob, cmap='gray')
plt.subplot(2, 2, 3)
plt.imshow(openinng_blob, cmap='gray')
plt.subplot(2, 2, 4)
plt.imshow(result-edge, cmap='gray')
plt.show()
# Step D: Get the borders of the circles
edges = cv.Canny(result, 100, 200)


cv.imshow('Circle Borders', edges)
cv.waitKey(0)
cv.destroyAllWindows()
