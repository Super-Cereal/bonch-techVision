import numpy as np


def findCoords(labels, minBoxSize, maxBoxSize):
    boxes = {}

    for label in np.unique(labels):
        ys, xs = np.where(labels == label)

        if minBoxSize <= ys.shape[0] <= maxBoxSize:
            boxes[label] = ((xs.min(), ys.min()), (xs.max(), ys.max()))

    return boxes
