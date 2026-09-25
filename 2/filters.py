import cv2 as cv
import numpy as np

from .convolution import convolution


def ru(scalarOrArray):
    return np.round(scalarOrArray).astype(np.uint8)


def lowPassFilter_my(image, kernel_size):
    assert kernel_size % 2 == 1
    kernel = np.ones((kernel_size, kernel_size), dtype=np.float64) / kernel_size**2

    return ru(convolution(image, kernel))


def lowPassFilter(image, kernel_size):
    return cv.blur(image, (kernel_size, kernel_size))


def highPassFilter_my(image):
    dx_kernel = np.array([[-1, 0, 1],
                          [-2, 0, 2],
                          [-1, 0, 1]])
    dy_kernel = np.array([[1, 2, 1],
                          [0, 0, 0],
                          [-1, -2, -1]])

    dx = convolution(image, dx_kernel)
    dy = convolution(image, dy_kernel)
    magnitude = np.sqrt(dx**2 + dy**2)

    return ru(magnitude), ru(dx), ru(dy)


def highPassFilter(image):
    dx = cv.Sobel(image, -1, dx=1, dy=0, ksize=3).astype(np.float32)
    dy = cv.Sobel(image, -1, dx=0, dy=1, ksize=3).astype(np.float32)
    magnitude = np.sqrt(dx**2 + dy**2)

    return ru(magnitude), ru(dx), ru(dy)
