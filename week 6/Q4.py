import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# Read in the input image in grayscale
img = cv.imread('chromophobe-renal-cell-carcinoma-6.jpg', cv.IMREAD_GRAYSCALE)

# Define a list to store the initial contour points
points = []

# Define a function to handle mouse click events and add clicked coordinates to the list of points
def click_event(event, x, y, flags, params):
    global points

    if event == cv.EVENT_LBUTTONDOWN:
        print(f'({x}, {y})')
        points.append([x, y])
        cv.circle(img, (x, y), 3, (0, 255, 255), -1)

# Create a window to display the image and set the mouse callback function
cv.namedWindow('Select Initial Contour')
cv.setMouseCallback('Select Initial Contour', click_event)

# Wait for the user to select points and display them on the image
while True:
    cv.imshow('Select Initial Contour', img)
    k = cv.waitKey(1) & 0xFF
    if k == 27: # ESC key
        break

# Create an energy function that will attach the initial contour to the edges of the object
energy_field = np.zeros_like(img)
for point in points:
    x, y = point
    energy_field[y, x] = 255

# Define a function to generate a new contour based on the previous contour and sampled points
def create_new_contour(previous_contour, sampled_points):
    new_contour = []
    for point in sampled_points:
        x, y = point
        window = energy_field[max(0, y-3):min(y+4, energy_field.shape[0]), max(0, x-3):min(x+4, energy_field.shape[1])]
        if not np.any(window):
            continue
        min_energy_x, min_energy_y = np.unravel_index(np.argmin(window), window.shape)
        new_point = [x + min_energy_y - 3, y + min_energy_x - 3]
        new_contour.append(new_point)
    return np.array(new_contour)

# Initialize the contour to the initial set of points
contour = np.array(points, dtype=np.int32)

# Iterate over the sampled points until the desired segmentation is achieved
for i in range(50):
    # Sample points from the current contour
    num_samples = 100
    step_size = int(len(contour) / num_samples)
    if step_size == 0:
        step_size = 1
    sampled_points = [contour[i] for i in range(0, len(contour), step_size)]

    # Create a new contour based on the previous contour and sampled points
    new_contour = create_new_contour(contour, sampled_points)

    # If the new contour is not significantly different from the previous contour, stop iterating
    if np.allclose(new_contour, contour, atol=1):
        break

    # Otherwise, update the contour and continue iterating
    contour = new_contour

# Draw the final contour on the image and display it
cv.polylines(img, [contour], True, (0, 255, 0), 2)
cv.imshow('Segmented Image', img)
cv.waitKey(0)
cv.destroyAllWindows()
