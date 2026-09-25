import numpy as np


def manualThreshold(image, t):
    result = np.copy(image)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            result[i, j] = 0 if image[i, j] <= t else 255

    return result
