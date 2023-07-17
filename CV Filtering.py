# -*- coding: utf-8 -*-
"""
Created on Sun Feb 12 07:02:08 2023

@author: patel
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np

# Read the image using OpenCV
img = cv2.imread("image.jpg")

plt.imshow(img)
plt.title("Original Image")
plt.show()

# Convert the image to grayscale
grayscale_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

plt.imshow(grayscale_image, cmap='gray')
plt.title("Grayscale Image")
plt.show()

# Convert the grayscale image to a 2-dimensional NumPy array
grayscale_image_array = np.array(grayscale_image)

# Define the filter
horizontal_filter = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
vertical_filter = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])


# Get the size of the image and filter
image_height, image_width = grayscale_image_array.shape
horizontal_filter_height, horizontal_filter_width = horizontal_filter.shape
vertical_filter_height, vertical_filter_width = vertical_filter.shape

# Create a zero array to store the convolved image
convolved = np.zeros(grayscale_image_array.shape)

# Perform the horizontal convolution
for i in range(image_height - horizontal_filter_height + 1):
    for j in range(image_width - horizontal_filter_width + 1):
        convolved[i, j] = np.sum(grayscale_image_array[i:i+horizontal_filter_height,
                                 j:j+horizontal_filter_width] * horizontal_filter)
plt.imshow(convolved, cmap='gray')
plt.title("Horizontally convolved Image")
plt.show()

# Perform the vertical convolution
for i in range(image_height - vertical_filter_height + 1):
    for j in range(image_width - vertical_filter_width + 1):
        convolved[i, j] = np.sum(grayscale_image_array[i:i+vertical_filter_height,
                                 j:j+vertical_filter_width] * vertical_filter)
plt.imshow(convolved, cmap='gray')
plt.title("Vertically convolved Image")
plt.show()


def gaussian_blur(image, kernel_size=3, sigma=1.0):
    # =============================================================================
    #     horizontal_filter = np.array([[-1,  0,  1], [-1,  0,  1], [-1,  0,  1]])
    #     vertical_filter = np.array([[-1,  -1,  1], [0,  0,  0], [1,  1,  1]])
    # =============================================================================

    horizontal_filter, vertical_filter = np.meshgrid(
        np.linspace(-1, 1, kernel_size), np.linspace(-1, 1, kernel_size))

    gaussian_function = np.exp(-((horizontal_filter **
                               2 + vertical_filter**2) / (2 * sigma**2)))
    gaussian_function = gaussian_function / gaussian_function.sum()

    # Perform the convolution
    convolution = np.zeros_like(image)
    for i in range(image.shape[0] - kernel_size + 1):
        for j in range(image.shape[1] - kernel_size + 1):
            convolution[i, j] = np.sum(
                image[i:i+kernel_size, j:j+kernel_size] * gaussian_function)
    return convolution


gaussian_blur_image = gaussian_blur(grayscale_image, 3, 1.0)

plt.imshow(gaussian_blur_image, cmap='gray')
plt.title("Gaussian Blur Image")
plt.show()
