import cv2
import numpy as np
# first we decide to using the right libraries

# here we read the image and it's size
img = cv2.imread('image.jpg')
height, width = img.shape[:2]

# defining the 2D matrix for rotation
M_rotate = cv2.getRotationMatrix2D((width/2, height/2), -30, 1)

# defining a matrix for changing the size of image to half
M_scale = np.float32([[0.5, 0, 0], [0, 0.5, 0]])

# combining the scale and rotation matrix
M_combined = np.matmul(M_scale, M_rotate)

# make a compound matrix for image
img_rotated_scaled = cv2.warpAffine(img, M_combined, (width, height))

# transforming the image using cv2.warpAffine() function
M_translate = np.float32([[1, 0, 0], [0, 1, 10]])
img_final = cv2.warpAffine(img_rotated_scaled, M_translate, (width, height))

# showing the final image we got here
cv2.imshow('Final Image', img_final)
cv2.waitKey(0)
cv2.destroyAllWindows()
