import cv2 as cv
import numpy as np


def ru(scalarOrArray):
    return np.round(scalarOrArray).astype(np.uint8)


def highPassFilter(image):
    dx = cv.Sobel(image, -1, dx=1, dy=0, ksize=3).astype(np.float32)
    dy = cv.Sobel(image, -1, dx=0, dy=1, ksize=3).astype(np.float32)
    magnitude = np.sqrt(dx**2 + dy**2)

    return ru(magnitude), ru(dx), ru(dy)
