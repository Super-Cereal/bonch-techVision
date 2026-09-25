import cv2 as cv
import numpy as np

img = cv.imread("./assets/image.png")

cv.imwrite("./assets/3_5.jpg", cv.add(img, 130.0)) # type: ignore
