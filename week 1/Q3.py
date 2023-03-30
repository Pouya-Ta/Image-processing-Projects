import numpy as np
import cv2


def normalize_image(image):

    # calculating the size
    height, width = image.shape

    # up and sown bound
    upper_bound = np.amax(image, axis=1)
    lower_bound = np.amin(image, axis=1)

    # normalizing
    for i in range(height):
        image[i, :] = (image[i, :] - lower_bound[i]) / (upper_bound[i] - lower_bound[i]) * 255

    # column mean calculation
    column_means = np.mean(image, axis=0)

    # row mean calculation
    row_means = np.mean(image, axis=1)

    return image.astype(np.uint8), column_means, row_means


# input image
img = cv2.imread('HW1_Q4_object_LE.bmp', cv2.IMREAD_GRAYSCALE)


normalized_image, column_means, row_means = normalize_image(img)

# making the first and end image
output_image = cv2.hconcat([img, normalized_image])

# showing the first and end image
cv2.imshow('Raw Image and Normalized Image', output_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# mid of anything we calculated
print('Column Means:')
print(column_means)
print('\nRow Means:')
print(row_means)
