import cv2 as cv
import numpy as np

from .histogram import histogram


def makeCdf(histogram):
    integral = np.zeros_like(histogram)
    integral[0] = histogram[0]

    for i in range(1, integral.shape[0]):
        integral[i] = integral[i-1] + histogram[i]

    return integral


def histEqualization(image):
    h = histogram(image)
    numPixels = image.shape[0] * image.shape[1]
    cdf = makeCdf(h)

    # normalization
    cdfMin = 0
    for el in h:
        if el != 0:
            cdfMin = el
            break

    sourceRange = numPixels
    destinationRange = 255
    lut = np.zeros_like(h)

    for i in range(destinationRange + 1):
        top = max(cdf[i] - cdfMin, 0)
        bottom = sourceRange - cdfMin

        lut[i] = (top / bottom) * (destinationRange)

    return cv.LUT(image, lut).astype(np.uint8)


def coloredHistEqualization(image):
    hsv = cv.cvtColor(image, cv.COLOR_RGB2HSV)
    h, s, v = cv.split(hsv)

    v_equalized = cv.equalizeHist(v)

    return cv.cvtColor(cv.merge((h, s, v_equalized)), cv.COLOR_HSV2RGB)


def coloredClaheHistEqualization(image, clipLimit=7.0):
    hsv = cv.cvtColor(image, cv.COLOR_RGB2HSV)
    h, s, v = cv.split(hsv)

    v_clahe = cv.createCLAHE(clipLimit, tileGridSize=(8, 8)).apply(v)

    return cv.cvtColor(cv.merge((h, s, v_clahe)), cv.COLOR_HSV2RGB)
