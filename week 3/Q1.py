import cv2
import numpy as np

def apply_spatial_filter(filter_type, kernel_size, input_image):
    """
    Apply spatial filter on the input image using the specified filter type and kernel size.

    Args:
        filter_type (str): Type of filter to be applied. Options are "averaging" or "median".
        kernel_size (tuple): Size of the kernel to be used for the filter.
        input_image (str): Name/Path of the input image file.

    Returns:
        output_image (numpy.ndarray): Output Image after applying the desired Spatial Filter
    """

    # Load input image
    img = cv2.imread(input_image)

    # Convert input image to grayscale
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply the specified spatial filter
    if filter_type == 'averaging':
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
        output_image = cv2.filter2D(gray_img, -1, kernel)
    elif filter_type == 'median':
        output_image = cv2.medianBlur(gray_img, kernel_size[0])

    return output_image


def apply_laplacian_filter(angle, input_image):
    """
    Apply isotropic Laplacian filter with specified angle on the input image.

    Args:
        angle (int): Angle in degrees at which to apply the Laplacian filter.
        input_image (str): Name/Path of the input image file.

    Returns:
        output_image (numpy.ndarray): Output Image after applying the Laplacian Filter
    """

    # Load input image
    img = cv2.imread(input_image)

    # Convert input image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Create Laplacian filter with specified angle
    kernel_size = 3
    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    mid = kernel_size // 2
    for i in range(kernel_size):
        for j in range(kernel_size):
            x, y = i - mid, j - mid
            if angle == 0:
                kernel[i, j] = -2*y*y + x*x
            elif angle == 45:
                kernel[i, j] = x*x - y*y + 2*x*y
            elif angle == 90:
                kernel[i, j] = -2*x*x + y*y
            elif angle == 135:
                kernel[i, j] = x*x - y*y - 2*x*y
            elif angle == 180:
                kernel[i, j] = -y*y

    # Apply Laplacian filter to input image
    laplacian = cv2.filter2D(gray, -1, kernel)

    # Normalize output image for display
    laplacian_norm = cv2.normalize(laplacian, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)

    return laplacian_norm

# Load input image
input_image = 'kidney.png'
img = cv2.imread(input_image)

# Apply averaging filter with 5x5 kernel
avg_filtered_img = apply_spatial_filter('averaging', (5,5), input_image)

# Convert to 3-channel image for concatenation
avg_filtered_img = cv2.cvtColor(avg_filtered_img, cv2.COLOR_GRAY2BGR)

# Check dimensions of filtered image and resize if necessary
if avg_filtered_img.shape != img.shape:
    avg_filtered_img = cv2.resize(avg_filtered_img, (img.shape[1], img.shape[0]))

# Apply median filter with 5x5 kernel
med_filtered_img = apply_spatial_filter('median', (5,5), input_image)

# Convert to 3-channel image for concatenation
med_filtered_img = cv2.cvtColor(med_filtered_img, cv2.COLOR_GRAY2BGR)

# Check dimensions of filtered image and resize if necessary
if med_filtered_img.shape != img.shape:
    med_filtered_img = cv2.resize(med_filtered_img, (img.shape[1], img.shape[0]))

# Display all three images side-by-side
concatenated_img1 = cv2.hconcat([img, avg_filtered_img, med_filtered_img])
cv2.imshow('Original vs Filtered Images', concatenated_img1)
cv2.waitKey(0)

# Apply Laplacian filter at 45 degree angle
laplacian_45 = apply_laplacian_filter(45, input_image)

# Convert to 3-channel image for concatenation
laplacian_45 = cv2.cvtColor(laplacian_45, cv2.COLOR_GRAY2BGR)

# Apply Laplacian filter at 90 degree angle
laplacian_90 = apply_laplacian_filter(90, input_image)

# Convert to 3-channel image for concatenation
laplacian_90 = cv2.cvtColor(laplacian_90, cv2.COLOR_GRAY2BGR)

# Apply Laplacian filter at 180 degree angle
laplacian_180 = apply_laplacian_filter(180, input_image)

# Convert to 3-channel image for concatenation
laplacian_180 = cv2.cvtColor(laplacian_180, cv2.COLOR_GRAY2BGR)

# Check dimensions of Laplacian filtered images and resize if necessary
if laplacian_45.shape != img.shape:
    laplacian_45 = cv2.resize(laplacian_45, (img.shape[1], img.shape[0]))
if laplacian_90.shape != img.shape:
    laplacian_90 = cv2.resize(laplacian_90, (img.shape[1], img.shape[0]))
if laplacian_180.shape != img.shape:
    laplacian_180 = cv2.resize(laplacian_180, (img.shape[1], img.shape[0]))

# Display all four images side-by-side
concatenated_img2 = cv2.hconcat([img, laplacian_45, laplacian_90, laplacian_180])
cv2.imshow('Original vs Laplacian Filtered Images', concatenated_img2)
cv2.waitKey(0)

cv2.destroyAllWindows()
