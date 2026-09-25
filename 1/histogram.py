import numpy as np


def histogram(img):
    h = np.zeros(256, dtype=np.int32)

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            intence = img[i, j]
            h[intence] = h[intence] + 1

    return h
