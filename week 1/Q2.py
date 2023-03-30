import cv2
import numpy as np
import time

# Load the input image.
img = cv2.imread('HW1_Q3.png')

# Define a scaling factor and number of frames.
scale_factor = 0.5
num_frames = 10

# Get the width and height of the frame and resize the image to match with it.
height, width, _ = img.shape
new_width = int(width * scale_factor)
new_height = int(height * scale_factor)
img = cv2.resize(img, (new_width, new_height))

# Define the transformation matrices for translations, scales, rotations, and shears.
translation_matrix = np.float32([[1, 0, -160], [0, 1, -160]])
scale_matrix = np.float32([[0.5, 0, 0], [0, 0.5, 0]])
rotation_matrix = cv2.getRotationMatrix2D((new_width / 2, new_height / 2), -50, 1)
shear_matrix = np.float32([[1, 0.2, 0], [0, 1, 0]])

# Define an array of matrices.
transformation_matrices = [translation_matrix, scale_matrix, rotation_matrix, shear_matrix]

# Define an empty array to store the final output.
out_frames = []

# Loop through each frame and generate the images.
for i in range(num_frames):

    # Create a copy of the original image.
    img_copy = img.copy()

    # Loop through each transformation matrix and apply it to the copy of the original image.
    for matrix in transformation_matrices:
        img_copy = cv2.warpAffine(img_copy, matrix, (new_width, new_height))

    # Concatenate the original image with the transformed image and store it in the final array.
    out_frame = np.concatenate((img, cv2.flip(img_copy, 1)), axis=1)
    out_frames.append(out_frame)

    # Wait for a second before generating the next frame.
    time.sleep(1)

# Combine the frames to create a video.
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video = cv2.VideoWriter('output.mp4', fourcc, 10, (2 * new_width, new_height))
for frame in out_frames:
    video.write(frame)

# Release the video writer and delete the temporary image files.
video.release()
cv2.destroyAllWindows()
