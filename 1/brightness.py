import cv2 as cv


def brightness(img, delta):
    r, g, b = cv.split(img)
    r = cv.add(r, delta)
    g = cv.add(g, delta)
    b = cv.add(b, delta)
    return cv.merge((r, g, b))


def brightness_2(img, delta):
    return cv.convertScaleAbs(img, beta=delta)
