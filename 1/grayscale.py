import numpy as np


def grayscaleFormula_1(R, G, B):
    return 0.299*R + 0.587*G + 0.114*B


def grayscaleFormula_2(R, G, B):
    return 0.2126*R + 0.7152*G + 0.0722*B


def grayscaleFormula_3(R, G, B):
    return 0.333*R + 0.333*G + 0.333*B


def grayscale(img, formula=1):
    w, h, _ = img.shape
    result = np.zeros((w, h), dtype=np.uint8)

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if formula == 1:
                result[i, j] = grayscaleFormula_1(*img[i, j])
            elif formula == 2:
                result[i, j] = grayscaleFormula_2(*img[i, j])
            elif formula == 3:
                result[i, j] = grayscaleFormula_3(*img[i, j])

    return result
