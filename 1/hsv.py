import cv2 as cv
import numpy as np


def hsv_delta(img, h_delta: int = 0, s_delta: int = 0):
    h, s, v = cv.split(cv.cvtColor(img, cv.COLOR_RGB2HSV))

    h = ((h.astype(np.int16) + h_delta) % 180).astype(np.uint8)
    s = np.clip(s.astype(np.int16) + s_delta, 0, 255).astype(np.uint8)

    hsv = cv.merge((h, s, v))
    return cv.cvtColor(hsv, cv.COLOR_HSV2RGB)