import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import random
from pathlib import Path

from .threshold import manualThreshold
from .filters import lowPassFilter, lowPassFilter_my, highPassFilter, highPassFilter_my
from .findCoords import findCoords

filename = './assets/i4.png'
binary_threshold_thresh = 50
adaptive_threshold_block_size = 165
adaptive_threshold_C = 6

if __name__ == '__main__':
    base_dir = Path(__file__).resolve().parent
    filename = base_dir / filename

    image_orig = cv.imread(filename)

    if image_orig is None:
        raise FileNotFoundError(f'Картинки {filename} нет!')

    image = cv.cvtColor(image_orig, cv.COLOR_BGR2GRAY)
    image_hist = cv.calcHist([image], [0], None, [256], [0, 256])

    # Бинаризация
    binary_threshold = manualThreshold(image, binary_threshold_thresh)
    binary_threshold_hist = cv.calcHist([binary_threshold], [0], None, [256], [0, 256])

    otsu_threshold_thresh, otsu_threshold = cv.threshold(image, 0, 255, type=cv.THRESH_BINARY + cv.THRESH_OTSU)
    otsu_threshold_hist = cv.calcHist([otsu_threshold], [0], None, [256], [0, 256])

    adaptive_mean_threshold = cv.adaptiveThreshold(
        image, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, adaptive_threshold_block_size, adaptive_threshold_C)
    adaptive_mean_threshold_hist = cv.calcHist([adaptive_mean_threshold], [0], None, [256], [0, 256])

    adaptive_gaussian_threshold = cv.adaptiveThreshold(
        image, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, adaptive_threshold_block_size, adaptive_threshold_C)
    adaptive_gaussian_threshold_hist = cv.calcHist([adaptive_gaussian_threshold], [0], None, [256], [0, 256])

    fig, ax = plt.subplots(2, 5, figsize=(18, 10))

    ax[0, 0].imshow(image, cmap='gray')
    ax[0, 0].axis("off")
    ax[0, 0].axis("off")
    ax[0, 0].set_title('Картинка изначальная')
    ax[1, 0].plot(image_hist)
    ax[1, 0].set_title('Гистограмма изначальная')

    ax[0, 1].imshow(binary_threshold, cmap='gray')
    ax[0, 1].axis("off")
    ax[0, 1].set_title('Картинка ручная\nбинаризация')
    ax[1, 1].plot(binary_threshold_hist)
    ax[1, 1].axvline(binary_threshold_thresh, color="red", label="Порог бинаризации")
    ax[1, 1].set_title('Гистограмма ручная\nбинаризация')
    ax[1, 1].legend()

    ax[0, 2].imshow(otsu_threshold, cmap='gray')
    ax[0, 2].axis("off")
    ax[0, 2].set_title('Картинка Отцу\nбинаризация')
    ax[1, 2].plot(otsu_threshold_hist)
    ax[1, 2].axvline(otsu_threshold_thresh, color="red", label="Порог бинаризации")
    ax[1, 2].set_title('Гистограмма Отцу\nбинаризация')
    ax[1, 2].legend()

    ax[0, 3].imshow(adaptive_mean_threshold, cmap='gray')
    ax[0, 3].axis("off")
    ax[0, 3].set_title('Картинка адаптивная\nсредняя бинаризация')
    ax[1, 3].plot(adaptive_mean_threshold_hist)
    ax[1, 3].set_title('Гистограмма адаптивная\nсредняя бинаризация')

    ax[0, 4].imshow(adaptive_gaussian_threshold, cmap='gray')
    ax[0, 4].axis("off")
    ax[0, 4].set_title('Картинка адаптивная\nгауссова бинаризация')
    ax[1, 4].plot(adaptive_gaussian_threshold_hist)
    ax[1, 4].set_title('Гистограмма адаптивная\nгауссова бинаризация')

    plt.tight_layout()
    plt.show()

    # Инвертация
    image_inverted = cv.bitwise_not(image)
    image_inverted_hist = cv.calcHist([image_inverted], [0], None, [256], [0, 256])

    adaptive_gaussian_threshold_inverted = cv.bitwise_not(adaptive_gaussian_threshold)
    adaptive_gaussian_threshold_inverted_hist = cv.calcHist(
        [adaptive_gaussian_threshold_inverted], [0], None, [256], [0, 256])

    fig, ax = plt.subplots(2, 4, figsize=(18, 10))

    ax[0, 0].imshow(image, cmap='gray')
    ax[0, 0].axis("off")
    ax[0, 0].set_title('Картинка изначальная')
    ax[1, 0].plot(image_hist)
    ax[1, 0].set_title('Гистограмма изначальная')

    ax[0, 1].imshow(image_inverted, cmap='gray')
    ax[0, 1].axis("off")
    ax[0, 1].set_title('Картинка инвертация')
    ax[1, 1].plot(image_inverted_hist)
    ax[1, 1].set_title('Гистограмма инвертация')

    ax[0, 2].imshow(adaptive_gaussian_threshold, cmap='gray')
    ax[0, 2].axis("off")
    ax[0, 2].set_title('Картинка адаптивная\nгауссова бинаризация')
    ax[1, 2].plot(adaptive_gaussian_threshold_hist)
    ax[1, 2].set_title('Гистограмма адаптивная\nгауссова бинаризация')

    ax[0, 3].imshow(adaptive_gaussian_threshold_inverted, cmap='gray')
    ax[0, 3].axis("off")
    ax[0, 3].set_title('Картинка инвертация\nадаптивная гауссова\nбинаризация')
    ax[1, 3].plot(adaptive_gaussian_threshold_inverted_hist)
    ax[1, 3].set_title('Гистограмма инвертация\nадаптивная гауссова\nбинаризация')

    plt.tight_layout()
    plt.show()

    # Фильтр низких частот
    def lowPassFilter_adapter(kernel_size, number, ax):
        # low_pass_filter_image = lowPassFilter_my(image, kernel_size)
        low_pass_filter_image = lowPassFilter(image, kernel_size)
        _, low_pass_filter_image_threshold = cv.threshold(
            low_pass_filter_image, 0, 255, type=cv.THRESH_BINARY + cv.THRESH_OTSU)

        ax[0, number].imshow(low_pass_filter_image, cmap='gray')
        ax[0, number].axis("off")
        ax[0, number].set_title(f'Картинка ФНЧ ({kernel_size})')
        ax[1, number].imshow(low_pass_filter_image_threshold, cmap='gray')
        ax[0, number].axis("off")
        ax[1, number].set_title('Картинка ФНЧ Отцу бинаризация')

    fig, ax = plt.subplots(2, 5, figsize=(18, 10))

    ax[0, 0].imshow(image, cmap='gray')
    ax[0, 0].axis("off")
    ax[0, 0].set_title('Картинка изначальная')
    ax[1, 0].set_visible(False)

    lowPassFilter_adapter(5, 1, ax)
    lowPassFilter_adapter(11, 2, ax)
    lowPassFilter_adapter(13, 3, ax)
    lowPassFilter_adapter(15, 4, ax)

    plt.tight_layout()
    plt.show()

    # Фильтр высоких частот
    fig, ax = plt.subplots(2, 4, figsize=(18, 10))

    ax[0, 0].imshow(image, cmap='gray')
    ax[0, 0].axis("off")
    ax[0, 0].set_title('Картинка изначальная')
    ax[1, 0].set_visible(False)

    # high_pass_filter_image, dx, dy = highPassFilter_my(image)
    high_pass_filter_image, dx, dy = highPassFilter(image)
    _, high_pass_filter_image_threshold = cv.threshold(
        high_pass_filter_image, 0, 255, type=cv.THRESH_BINARY + cv.THRESH_OTSU)
    _, dx_threshold = cv.threshold(dx, 0, 255, type=cv.THRESH_BINARY + cv.THRESH_OTSU)
    _, dy_threshold = cv.threshold(dy, 0, 255, type=cv.THRESH_BINARY + cv.THRESH_OTSU)

    ax[0, 1].imshow(high_pass_filter_image, cmap='gray')
    ax[0, 1].axis("off")
    ax[0, 1].set_title(f'Картинка ФВЧ')
    ax[1, 1].imshow(high_pass_filter_image_threshold, cmap='gray')
    ax[1, 1].axis("off")
    ax[1, 1].set_title('Картинка ФВЧ Отцу бинаризация')

    ax[0, 2].imshow(dx, cmap='gray')
    ax[0, 2].axis("off")
    ax[0, 2].set_title(f'Картинка ФВЧ по X')
    ax[1, 2].imshow(dx_threshold, cmap='gray')
    ax[1, 2].axis("off")
    ax[1, 2].set_title('Картинка ФВЧ по X Отцу бинаризация')

    ax[0, 3].imshow(dy, cmap='gray')
    ax[0, 0].axis("off")
    ax[0, 3].set_title(f'Картинка ФВЧ по Y')
    ax[1, 3].imshow(dy_threshold, cmap='gray')
    ax[1, 3].axis("off")
    ax[1, 3].set_title('Картинка ФВЧ по Y Отцу бинаризация')

    plt.tight_layout()
    plt.show()

    # Распознавание
    image_gray = cv.cvtColor(image_orig, cv.COLOR_BGR2GRAY)
    image_blur = cv.GaussianBlur(image_gray, (9, 9), 0)
    image_binarized = cv.adaptiveThreshold(
        image_blur, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 215, 20)
    image_binarized = cv.bitwise_not(image_binarized)

    def rgb_random():
        r = np.round(random.random() * 255)
        g = np.round(random.random() * 255)
        b = np.round(random.random() * 255)
        return (r, g, b)

    image_labelled_num_labels, image_labelled_labeles = cv.connectedComponents(image_binarized)
    image_labelled_labels_color = np.array([rgb_random() for _ in range(image_labelled_num_labels)], dtype=np.uint8)
    image_labelled_labels_color[0] = (0, 0, 0)
    image_labelled_colored = image_labelled_labels_color[image_labelled_labeles]  # type: ignore

    areas = np.bincount(image_labelled_labeles.ravel(), minlength=image_labelled_num_labels)  # type: ignore
    areas = np.sort(areas)
    print("Размеры выделенных меток: ", areas)

    boxes = findCoords(image_labelled_labeles, 30, 300)
    image_with_boxes = np.copy(image_orig)
    for label in boxes:
        box = boxes[label]
        image_with_boxes = cv.rectangle(image_with_boxes, box[0], box[1], color=(255, 0, 0))

   # --- Вывод всех картинок в одном окне через plt.subplots ---
    images = [
        ("Картинка оригинальная", cv.cvtColor(image_orig, cv.COLOR_BGR2RGB)),
        ("Картинка серая", image_gray),
        ("Картинка серая блюр", image_blur),
        ("Картинка серая блюр\nбинаризированная", image_binarized),
        ("Картинка серая блюр\nбинаризированная\nс найденными обьектами", image_labelled_colored),
        ("Картинка оригинальная с отмеченными обьектами", cv.cvtColor(image_with_boxes, cv.COLOR_BGR2RGB)),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.ravel()

    for ax, (title, img) in zip(axes, images):
        if img.ndim == 2:
            ax.imshow(img, cmap="gray")
        else:
            ax.imshow(img)
        ax.set_title(title)
        ax.axis("off")

    plt.tight_layout()
    plt.show()
