import cv2 as cv
import matplotlib.pyplot as plt
from pathlib import Path

from .grayscale import grayscale
from .histogram import histogram
from .equalizing import histEqualization, makeCdf, coloredHistEqualization, coloredClaheHistEqualization
from .brightness import brightness
from .hsv import hsv_delta


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent

    under_exposed = cv.cvtColor(cv.imread(base_dir / "assets/1.jpg"), cv.COLOR_BGR2RGB)  # type: ignore
    normally_exposed = cv.cvtColor(cv.imread(base_dir / "assets/2.jpg"), cv.COLOR_BGR2RGB)  # type: ignore
    pre_exposed = cv.cvtColor(cv.imread(base_dir / "assets/3_5.jpg"), cv.COLOR_BGR2RGB)  # type: ignore

    images = [
        under_exposed,
        normally_exposed,
        pre_exposed
    ]

    for image_rgb in images:
        # преобразование в серый и гистограммы
        image_gray = cv.cvtColor(image_rgb, cv.COLOR_RGB2GRAY)
        image_gray_hist = cv.calcHist([image_gray], [0], None, [256], [0, 256])
        image_gray_hist_2 = histogram(image_gray)

        image_gray_2 = grayscale(image_rgb, formula=1)
        image_gray_2_hist = cv.calcHist([image_gray_2], [0], None, [256], [0, 256])
        image_gray_2_hist_2 = histogram(image_gray_2)

        # print(image_rgb.dtype, image_rgb.shape, image_rgb.min(), image_rgb.max())
        # print(image_gray.dtype, image_gray.shape, image_gray.min(), image_gray.max())
        # print(image_gray_2.dtype, image_gray_2.shape, image_gray_2.min(), image_gray_2.max())

        fig, ax = plt.subplots(3, 3, figsize=(14, 8))

        ax[0, 0].imshow(image_rgb)
        ax[0, 0].axis(False)
        ax[0, 0].set_title('Картинка RGB')
        ax[0, 1].set_visible(False)
        ax[0, 2].set_visible(False)

        ax[1, 0].imshow(image_gray, cmap='gray')
        ax[1, 0].axis(False)
        ax[1, 0].set_title('Grayscale OpenCV,\nКартинка')
        ax[1, 1].plot(image_gray_hist)
        ax[1, 1].set_title('Grayscale OpenCV,\nГистограмма OpenCV')
        ax[1, 2].plot(image_gray_hist_2)
        ax[1, 2].set_title('Grayscale OpenCV,\nГистограмма ручная')

        ax[2, 0].imshow(image_gray_2, cmap='gray')
        ax[2, 0].axis(False)
        ax[2, 0].set_title('Grayscale ручное,\nКартинка')
        ax[2, 1].plot(image_gray_2_hist)
        ax[2, 1].set_title('Grayscale ручное,\nГистограмма OpenCV')
        ax[2, 2].plot(image_gray_2_hist_2)
        ax[2, 2].set_title('Grayscale ручное,\nГистограмма ручная')

        plt.tight_layout()
        plt.show()

        # эквалайзинг
        image_gray_cdf = makeCdf(image_gray_hist)

        image_gray_equalized = cv.equalizeHist(image_gray)
        image_gray_equalized_hist = cv.calcHist([image_gray_equalized], [0], None, [256], [0, 256])
        image_gray_equalized_cdf = makeCdf(image_gray_equalized_hist)

        image_gray_equalized_2 = histEqualization(image_gray)
        image_gray_equalized_2_hist = cv.calcHist([image_gray_equalized_2], [0], None, [256], [0, 256])
        image_gray_equalized_2_cdf = makeCdf(image_gray_equalized_2_hist)

        fig, ax = plt.subplots(3, 3, figsize=(14, 8))

        ax[0, 0].imshow(image_gray, cmap='gray')
        ax[0, 0].axis(False)
        ax[0, 0].set_title('Картинка Grayscale')
        ax[0, 1].plot(image_gray_hist)
        ax[0, 1].set_title('Гистограмма Grayscale')
        ax[0, 2].plot(image_gray_cdf)
        ax[0, 2].set_title('Функция распределения Grayscale')

        ax[1, 0].imshow(image_gray_equalized, cmap='gray')
        ax[1, 0].axis(False)
        ax[1, 0].set_title('Картинка Grayscale,\nequalized OpenCV')
        ax[1, 1].plot(image_gray_equalized_hist)
        ax[1, 1].set_title('Гистограмма Grayscale,\nequalized OpenCV')
        ax[1, 2].plot(image_gray_equalized_cdf)
        ax[1, 2].set_title('Функция распределения Grayscale\nequalized OpenCV')

        ax[2, 0].imshow(image_gray_equalized_2, cmap='gray')
        ax[2, 0].axis(False)
        ax[2, 0].set_title('Картинка Grayscale,\nequalized ручное')
        ax[2, 1].plot(image_gray_equalized_2_hist)
        ax[2, 1].set_title('Гистограмма Grayscale,\nequalized ручное')
        ax[2, 2].plot(image_gray_equalized_2_cdf)
        ax[2, 2].set_title('Функция распределения Grayscal\nequalized ручное')

        plt.tight_layout()
        plt.show()

    # регулировка яркости
    fig, ax = plt.subplots(4, 4, figsize=(18, 10))

    ax[0, 0].imshow(under_exposed)
    ax[0, 0].axis(False)
    ax[0, 0].set_title('Картинка недо')
    ax[1, 0].plot(cv.calcHist([under_exposed], [0], None, [256], [0, 256]))
    ax[1, 0].set_title('Гистограмма недо (RED)')
    ax[2, 0].plot(cv.calcHist([under_exposed], [1], None, [256], [0, 256]))
    ax[2, 0].set_title('Гистограмма недо (GREEN)')
    ax[3, 0].plot(cv.calcHist([under_exposed], [2], None, [256], [0, 256]))
    ax[3, 0].set_title('Гистограмма недо (BLUE)')

    under_exposed_2 = brightness(under_exposed, delta=80)
    ax[0, 1].imshow(under_exposed_2)
    ax[0, 1].axis(False)
    ax[0, 1].set_title('Картинка яркость')
    ax[1, 1].plot(cv.calcHist([under_exposed_2], [0], None, [256], [0, 256]))
    ax[1, 1].set_title('Гистограмма яркость (RED)')
    ax[2, 1].plot(cv.calcHist([under_exposed_2], [1], None, [256], [0, 256]))
    ax[2, 1].set_title('Гистограмма яркость (GREEN)')
    ax[3, 1].plot(cv.calcHist([under_exposed_2], [2], None, [256], [0, 256]))
    ax[3, 1].set_title('Гистограмма яркость (BLUE)')

    under_exposed_3 = coloredHistEqualization(under_exposed)
    ax[0, 2].imshow(under_exposed_3)
    ax[0, 2].axis(False)
    ax[0, 2].set_title('Картинка эквализированная')
    ax[1, 2].plot(cv.calcHist([under_exposed_3], [0], None, [256], [0, 256]))
    ax[1, 2].set_title('Гистограмма эквализированная (RED)')
    ax[2, 2].plot(cv.calcHist([under_exposed_3], [1], None, [256], [0, 256]))
    ax[2, 2].set_title('Гистограмма эквализированная (GREEN)')
    ax[3, 2].plot(cv.calcHist([under_exposed_3], [2], None, [256], [0, 256]))
    ax[3, 2].set_title('Гистограмма эквализированная (BLUE)')

    under_exposed_4 = coloredClaheHistEqualization(under_exposed)
    ax[0, 3].imshow(under_exposed_4)
    ax[0, 3].axis(False)
    ax[0, 3].set_title('Картинка эквализированнае CLAHE')
    ax[1, 3].plot(cv.calcHist([under_exposed_4], [0], None, [256], [0, 256]))
    ax[1, 3].set_title('Гистограмма эквализированнае CLAHE (RED)')
    ax[2, 3].plot(cv.calcHist([under_exposed_4], [1], None, [256], [0, 256]))
    ax[2, 3].set_title('Гистограмма эквализированнае CLAHE (GREEN)')
    ax[3, 3].plot(cv.calcHist([under_exposed_4], [2], None, [256], [0, 256]))
    ax[3, 3].set_title('Гистограмма эквализированнае CLAHE (BLUE)')

    plt.tight_layout()
    plt.show()

    fig, ax = plt.subplots(4, 4, figsize=(18, 10))

    ax[0, 0].imshow(pre_exposed)
    ax[0, 0].axis(False)
    ax[0, 0].set_title('Картинка недо')
    ax[1, 0].plot(cv.calcHist([pre_exposed], [0], None, [256], [0, 256]))
    ax[1, 0].set_title('Гистограмма недо (RED)')
    ax[2, 0].plot(cv.calcHist([pre_exposed], [1], None, [256], [0, 256]))
    ax[2, 0].set_title('Гистограмма недо (GREEN)')
    ax[3, 0].plot(cv.calcHist([pre_exposed], [2], None, [256], [0, 256]))
    ax[3, 0].set_title('Гистограмма недо (BLUE)')

    pre_exposed_2 = brightness(pre_exposed, delta=-100)
    ax[0, 1].imshow(pre_exposed_2)
    ax[0, 1].axis(False)
    ax[0, 1].set_title('Картинка яркость')
    ax[1, 1].plot(cv.calcHist([pre_exposed_2], [0], None, [256], [0, 256]))
    ax[1, 1].set_title('Гистограмма яркость (RED)')
    ax[2, 1].plot(cv.calcHist([pre_exposed_2], [1], None, [256], [0, 256]))
    ax[2, 1].set_title('Гистограмма яркость (GREEN)')
    ax[3, 1].plot(cv.calcHist([pre_exposed_2], [2], None, [256], [0, 256]))
    ax[3, 1].set_title('Гистограмма яркость (BLUE)')

    pre_exposed_3 = coloredHistEqualization(pre_exposed)
    ax[0, 2].imshow(pre_exposed_3)
    ax[0, 2].axis(False)
    ax[0, 2].set_title('Картинка эквализированная')
    ax[1, 2].plot(cv.calcHist([pre_exposed_3], [0], None, [256], [0, 256]))
    ax[1, 2].set_title('Гистограмма эквализированная (RED)')
    ax[2, 2].plot(cv.calcHist([pre_exposed_3], [1], None, [256], [0, 256]))
    ax[2, 2].set_title('Гистограмма эквализированная (GREEN)')
    ax[3, 2].plot(cv.calcHist([pre_exposed_3], [2], None, [256], [0, 256]))
    ax[3, 2].set_title('Гистограмма эквализированная (BLUE)')

    pre_exposed_4 = coloredClaheHistEqualization(pre_exposed, clipLimit=30.0)
    ax[0, 3].imshow(pre_exposed_4)
    ax[0, 3].axis(False)
    ax[0, 3].set_title('Картинка эквализированнае с помощью CLAHE')
    ax[1, 3].plot(cv.calcHist([pre_exposed_4], [0], None, [256], [0, 256]))
    ax[1, 3].set_title('Гистограмма эквализированная CLAHE (RED)')
    ax[2, 3].plot(cv.calcHist([pre_exposed_4], [1], None, [256], [0, 256]))
    ax[2, 3].set_title('Гистограмма эквализированная CLAHE (GREEN)')
    ax[3, 3].plot(cv.calcHist([pre_exposed_4], [2], None, [256], [0, 256]))
    ax[3, 3].set_title('Гистограмма эквализированная CLAHE (BLUE)')

    plt.tight_layout()
    plt.show()

    # цвет и насыщенность
    fig, ax = plt.subplots(4, 3, figsize=(18, 10))

    ax[0, 0].imshow(normally_exposed)
    ax[0, 0].axis(False)
    ax[0, 0].set_title('Картинка норм')
    ax[1, 0].plot(cv.calcHist([normally_exposed], [0], None, [256], [0, 256]))
    ax[1, 0].set_title('Гистограмма норм (RED)')
    ax[2, 0].plot(cv.calcHist([normally_exposed], [1], None, [256], [0, 256]))
    ax[2, 0].set_title('Гистограмма норм (GREEN)')
    ax[3, 0].plot(cv.calcHist([normally_exposed], [2], None, [256], [0, 256]))
    ax[3, 0].set_title('Гистограмма норм (BLUE)')

    normally_exposed_2 = hsv_delta(normally_exposed, s_delta=80)
    ax[0, 1].imshow(normally_exposed_2)
    ax[0, 1].axis(False)
    ax[0, 1].set_title('Картинка насыщенность')
    ax[1, 1].plot(cv.calcHist([normally_exposed_2], [0], None, [256], [0, 256]))
    ax[1, 1].set_title('Гистограмма насыщенность (RED)')
    ax[2, 1].plot(cv.calcHist([normally_exposed_2], [1], None, [256], [0, 256]))
    ax[2, 1].set_title('Гистограмма насыщенность (GREEN)')
    ax[3, 1].plot(cv.calcHist([normally_exposed_2], [2], None, [256], [0, 256]))
    ax[3, 1].set_title('Гистограмма насыщенность (BLUE)')

    normally_exposed_3 = hsv_delta(normally_exposed, h_delta=90)
    ax[0, 2].imshow(normally_exposed_3)
    ax[0, 2].axis(False)
    ax[0, 2].set_title('Картинка цвет')
    ax[1, 2].plot(cv.calcHist([normally_exposed_3], [0], None, [256], [0, 256]))
    ax[1, 2].set_title('Гистограмма цвет (RED)')
    ax[2, 2].plot(cv.calcHist([normally_exposed_3], [1], None, [256], [0, 256]))
    ax[2, 2].set_title('Гистограмма насыщенность (GREEN)')
    ax[3, 2].plot(cv.calcHist([normally_exposed_3], [2], None, [256], [0, 256]))
    ax[3, 2].set_title('Гистограмма насыщенность (BLUE)')

    plt.tight_layout()
    plt.show()
