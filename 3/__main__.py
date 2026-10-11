import cv2 as cv
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

from .filters import highPassFilter

filename = './assets/2.png'


def show(ax, image, title, bitwise_not=False):
    if (bitwise_not):
        image = cv.bitwise_not(image)

    ax.imshow(image, cmap='gray')
    ax.axis("off")
    ax.set_title(title)


if __name__ == '__main__':
    base_dir = Path(__file__).resolve().parent
    filename = base_dir / filename

    image_orig_orig = cv.imread(filename)

    if image_orig_orig is None:
        raise FileNotFoundError(f'Картинки {filename} нет!')

    image_orig = cv.cvtColor(image_orig_orig, cv.COLOR_BGR2GRAY)
    image = cv.GaussianBlur(image_orig, (13, 13), 0)

    # 1
    # _, binary_threshold = cv.threshold(image, 90, 255, type=cv.THRESH_BINARY)
    # _, otsu_threshold = cv.threshold(image, 0, 255, type=cv.THRESH_BINARY + cv.THRESH_OTSU)

    # adaptive_mean_threshold = cv.adaptiveThreshold(
    #     image, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 7, 3)
    # adaptive_gaussian_threshold = cv.adaptiveThreshold(
    #     image, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 3)

    # highPassFilter_image = cv.GaussianBlur(image_orig, (7, 7), 0)
    # highPassFilter_image, _, _ = highPassFilter(highPassFilter_image)

    # fig, ax = plt.subplots(2, 3, figsize=(10, 10))

    # show(ax[0, 0], image, 'Картинка изначальная')
    # show(ax[0, 1], binary_threshold, 'Картинка бинаризация', True)
    # show(ax[0, 2], otsu_threshold, 'Картинка Отцу\nбинаризация', True)
    # show(ax[1, 0], adaptive_mean_threshold, 'Картинка адаптивная\nсредняя бинаризация', True)
    # show(ax[1, 1], adaptive_gaussian_threshold, 'Картинка адаптивная\nгауссова бинаризация', True)
    # show(ax[1, 2], highPassFilter_image, 'Фильтр высоких частот', False)

    # plt.tight_layout()
    # plt.show()

    # 2
    # fig, ax = plt.subplots(2, 3, figsize=(10, 10))
    # show(ax[0, 0], image, 'Картинка изначальная')

    # pixel_values = image.reshape((-1, 3)).astype(np.float32)
    # criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 1)
    # attempts = 10
    # flags = cv.KMEANS_PP_CENTERS

    # # Выполнение kmeans
    # compactness, labels, centers = cv.kmeans(pixel_values, 2, None, criteria, attempts, flags)
    # centers = np.uint8(centers)
    # segmented_image = centers[labels.flatten()].reshape(image.shape)
    # show(ax[0, 1], segmented_image, 'kmeans 2')

    # compactness, labels, centers = cv.kmeans(pixel_values, 3, None, criteria, attempts, flags)
    # centers = np.uint8(centers)
    # segmented_image = centers[labels.flatten()].reshape(image.shape)
    # show(ax[0, 2], segmented_image, 'kmeans 3')

    # compactness, labels, centers = cv.kmeans(pixel_values, 5, None, criteria, attempts, flags)
    # centers = np.uint8(centers)
    # segmented_image = centers[labels.flatten()].reshape(image.shape)
    # show(ax[1, 0], segmented_image, 'kmeans 5')

    # compactness, labels, centers = cv.kmeans(pixel_values, 6, None, criteria, attempts, flags)
    # centers = np.uint8(centers)
    # segmented_image = centers[labels.flatten()].reshape(image.shape)
    # show(ax[1, 1], segmented_image, 'kmeans 6')

    # compactness, labels, centers = cv.kmeans(pixel_values, 10, None, criteria, attempts, flags)
    # centers = np.uint8(centers)
    # segmented_image = centers[labels.flatten()].reshape(image.shape)
    # show(ax[1, 2], segmented_image, 'kmeans 10')

    # plt.tight_layout()
    # plt.show()

    # Выполнение meanShift
    kmeans_image = cv.pyrMeanShiftFiltering(image_orig_orig, sp=20, sr=40, maxLevel=2)

    # Если читали через cv.imread (BGR), а показываете через matplotlib:
    image_rgb    = cv.cvtColor(image_orig_orig, cv.COLOR_BGR2RGB)
    shifted_rgb  = cv.cvtColor(kmeans_image, cv.COLOR_BGR2RGB)

    fig, ax = plt.subplots(2, 3, figsize=(10, 10))
    show(ax[0, 0], image_rgb, 'Картинка изначальная')
    show(ax[0, 1], shifted_rgb, 'MeanShift')

    plt.tight_layout()
    plt.show()
