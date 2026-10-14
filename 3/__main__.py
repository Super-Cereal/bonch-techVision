import cv2 as cv
import time
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from sklearn.cluster import MeanShift, estimate_bandwidth

from .filters import highPassFilter

filename = './assets/2.png'
figsize = (17, 10)


if __name__ == '__main__':
    base_dir = Path(__file__).resolve().parent
    filename = base_dir / filename

    image = cv.imread(filename)
    assert image is not None

    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

    def appendPositions(image):
        result = np.zeros(shape=(image.shape[0], image.shape[1], image.shape[2] + 2))
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                result[i, j] = [(i / image.shape[0] * 255), (j / image.shape[1] * 255), *image[i, j]]

        return result

    data_with_positions = appendPositions(image)

    def showRGB(ax, image, title):
        ax.imshow(image)
        ax.axis("off")
        ax.set_title(title)

    def showGray(ax, image, title):
        ax.imshow(image, cmap='gray')
        ax.axis("off")
        ax.set_title(title)

    def showKMeansWithoutPositions(ax, result, title):
        compactness, labels, centers = result
        centers = np.uint8(centers)
        segmented_image = centers[labels.flatten()].reshape(image.shape)  # type: ignore

        ax.imshow(segmented_image)
        ax.axis("off")
        ax.set_title(title)

    def showKMeansWithPositions(ax, result, title):
        compactness, labels, centers = result
        centers = np.uint8(centers)
        segmented_image = centers[labels.flatten()].reshape(data_with_positions.shape)[:, :, 2:5]

        ax.imshow(segmented_image)
        ax.axis("off")
        ax.set_title(title)

    # # 1.1
    image_gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
    image_gray_blur = cv.GaussianBlur(image_gray, (9, 9), 0)
    # fig, ax = plt.subplots(2, 4, figsize=figsize)
    # fig.suptitle('Бинаризация черно белого', fontsize=16)
    # ax[0, 0].imshow(image)
    # ax[0, 0].axis("off")
    # ax[0, 0].set_title('Картинка изначальная')
    # showGray(ax[0, 1], cv.threshold(image_gray_blur, 90, 255, type=cv.THRESH_BINARY)[1], 'Картинка бинаризация')
    # showGray(ax[0, 2], cv.threshold(image_gray_blur, 0, 255, type=cv.THRESH_BINARY +
    #         cv.THRESH_OTSU)[1], 'Картинка Отцу\nбинаризация')
    # showGray(ax[1, 0], cv.adaptiveThreshold(
    #     image_gray_blur, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 7, 3), 'Картинка адаптивная\nсредняя бинаризация')
    # showGray(ax[1, 1], cv.adaptiveThreshold(
    #     image_gray_blur, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 3), 'Картинка адаптивная\nгауссова бинаризация')
    # showGray(ax[1, 2], highPassFilter(cv.GaussianBlur(image_gray, (7, 7), 0))[0], 'Фильтр высоких частот')

    # plt.tight_layout()
    # plt.show()

    # # 1.2 нейрокод
    # def highPassFilter(image, sigma=3):
    #     """Возвращает [high, low] — высокочастотную (границы) и низкочастотную (фон) составляющие."""
    #     low  = cv.GaussianBlur(image, (0, 0), sigma)
    #     high = cv.subtract(image, low)
    #     high = cv.normalize(high, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)
    #     return high, low

    # def kmeans_color_segmentation(image_bgr, features, K=5):
    #     """
    #     image_bgr : исходное цветное (H,W,3) BGR
    #     features  : (H*W, C) float32 — признаки для кластеризации
    #     """
    #     H, W = image_bgr.shape[:2]
    #     criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.5)
    #     _, labels, centers = cv.kmeans(features, K, None, criteria, 10,
    #                                 cv.KMEANS_PP_CENTERS)
    #     labels = labels.flatten()

    #     # средний цвет каждого кластера по исходной картинке
    #     palette = np.zeros((K, 3), np.float32)
    #     flat_img = image_bgr.reshape(-1, 3).astype(np.float32)
    #     for k in range(K):
    #         m = labels == k
    #         palette[k] = flat_img[m].mean(axis=0) if m.any() else 0

    #     seg = palette[labels].reshape(H, W, 3).astype(np.uint8)
    #     return seg, labels.reshape(H, W), centers

    # image_gray      = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
    # image_gray_blur = cv.GaussianBlur(image_gray, (9, 9), 0)

    # fig, ax = plt.subplots(2, 4, figsize=(16, 8))
    # fig.suptitle('Частоты → границы → цветная сегментация', fontsize=16)

    # ax[0, 0].imshow(image); ax[0, 0].axis("off")
    # ax[0, 0].set_title('Исходное')

    # showGray(ax[0, 1], cv.threshold(image_gray_blur, 90, 255, cv.THRESH_BINARY)[1],
    #         'Бинаризация 90')
    # showGray(ax[0, 2], cv.threshold(image_gray_blur, 0, 255,
    #         cv.THRESH_BINARY + cv.THRESH_OTSU)[1], 'Otsu')
    # showGray(ax[1, 0], cv.adaptiveThreshold(image_gray_blur, 255,
    #         cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 7, 3), 'Адаптивная')

    # # --- частотное разложение ---
    # high, low = highPassFilter(image_gray_blur, sigma=3)
    # showGray(ax[1, 1], high, 'Высокие частоты')

    # sharpened    = cv.addWeighted(low, 1.0, high, 0.7, 0)
    # edges        = cv.Canny(high, 40, 120)
    # kernel       = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
    # edges_closed = cv.morphologyEx(edges, cv.MORPH_CLOSE, kernel, iterations=2)
    # showGray(ax[1, 2], edges_closed, 'Границы (замкнутые)')

    # # --- K-means: признаки = unsharp + edges + цвет (Lab a,b) ---
    # lab      = cv.cvtColor(image, cv.COLOR_RGB2LAB)
    # chroma   = lab[:, :, 1:].astype(np.float32)                      # (H,W,2)

    # feats = np.dstack([
    #     sharpened.astype(np.float32) / 255.0,
    #     edges_closed.astype(np.float32) / 255.0,
    #     chroma[:, :, 0] / 255.0,
    #     chroma[:, :, 1] / 255.0,
    # ]).reshape(-1, 4).astype(np.float32)

    # # нормировка, чтобы каналы были сопоставимы
    # feats = (feats - feats.mean(0)) / (feats.std(0) + 1e-6)

    # K = 10
    # criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.5)
    # _, labels, _ = cv.kmeans(feats, K, None, criteria, 10, cv.KMEANS_PP_CENTERS)
    # labels = labels.reshape(sharpened.shape)

    # # цветная визуализация по среднему цвету кластера
    # image_bgr = cv.cvtColor(image, cv.COLOR_RGB2BGR)
    # seg_bgr, _, _ = kmeans_color_segmentation(image_bgr, feats, K=K)
    # seg_rgb = cv.cvtColor(seg_bgr, cv.COLOR_BGR2RGB)

    # ax[1, 3].imshow(seg_rgb)
    # ax[1, 3].set_title(f'Цветная сегментация (K={K})')
    # ax[1, 3].axis("off")

    # plt.tight_layout()
    # plt.show()

    # # 2.1 Выполнение kmeans только по цвету
    # with ProcessPoolExecutor(max_workers=10) as ex:
    #     data_one_line = image.reshape((-1, 3)).astype(np.float32)
    #     criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.5)
    #     attempts = 10
    #     flags = cv.KMEANS_PP_CENTERS

    #     ks = [3, 5, 10, 15, 20]
    #     futures = []
    #     for k in ks:
    #         futures.append(ex.submit(cv.kmeans, data_one_line, k, None, criteria, attempts, flags))  # type: ignore

    #     fig, ax = plt.subplots(2, 3, figsize=figsize)
    #     fig.suptitle('kmeans только по цвету', fontsize=16)
    #     showRGB(ax[0, 0], image, 'Картинка изначальная')
    #     axes = [ax[0, 1], ax[0, 2], ax[1, 0], ax[1, 1], ax[1, 2]]
    #     for ax, future, k in zip(axes, futures, ks):
    #         showKMeansWithoutPositions(ax, future.result(), f'clusters {k}')
    #     plt.tight_layout()
    #     plt.show()

    # # 2.2 Выполнение kmeans по цвету и позиции
    # with ProcessPoolExecutor(max_workers=10) as ex:
    #     data_one_line = data_with_positions.reshape((-1, 5)).astype(np.float32)
    #     criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.5)
    #     attempts = 10
    #     flags = cv.KMEANS_PP_CENTERS
    #     ks = [3, 5, 10, 15, 20, 25, 30]
    #     futures = []
    #     for k in ks:
    #         futures.append(ex.submit(cv.kmeans, data_one_line, k, None, criteria, attempts, flags))  # type: ignore

    #     fig, ax = plt.subplots(2, 4, figsize=figsize)
    #     fig.suptitle('kmeans по цвету и позиции', fontsize=16)
    #     showRGB(ax[0, 0], image, 'Картинка изначальная')
    #     axes = [ax[0, 1], ax[0, 2], ax[0, 3], ax[1, 0], ax[1, 1], ax[1, 2], ax[1, 3]]
    #     for ax, future, k in zip(axes, futures, ks):
    #         showKMeansWithPositions(ax, future.result(), f'clusters {k}')
    #     plt.tight_layout()
    #     plt.show()

    # # 2.3 Выполнение opencv.pyrMeanShiftFiltering по цвету и позиции
    # with ProcessPoolExecutor(max_workers=10) as ex:
    #     SP = np.array([10,  15,  20,  25,  30])
    #     SR = SP * 2
    #     MAX_LEVEL = [1,   1,   2,   2,   3]
    #     futures = []
    #     for sp, sr, maxLevel in zip(SP, SR, MAX_LEVEL):
    #         futures.append(ex.submit(cv.pyrMeanShiftFiltering, image, sp=sp, sr=sr, maxLevel=maxLevel))

    #     fig, ax = plt.subplots(2, 3, figsize=(15, 10))
    #     fig.suptitle('opencv.pyrMeanShiftFiltering', fontsize=16)
    #     showRGB(ax[0, 0], image, 'Картинка изначальная')
    #     axes = [ax[0, 1], ax[0, 2], ax[1, 0], ax[1, 1], ax[1, 2]]
    #     for ax, future, sp, sr, maxLevel in zip(axes, futures, SP, SR, MAX_LEVEL):
    #         showRGB(ax, future.result(), f'sp {sp}, sr {sr}, maxLevel {maxLevel} ')
    #     plt.tight_layout()
    #     plt.show()

    #     criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.5)
    #     K = 20
    #     attempts = 10
    #     flags = cv.KMEANS_PP_CENTERS
    #     futures2 = []
    #     for future in futures:
    #         img = appendPositions(future.result()).reshape((-1, 5)).astype(np.float32)
    #         futures2.append(ex.submit(cv.kmeans, img, K, None, criteria, attempts, flags))  # type: ignore

    #     fig, ax = plt.subplots(2, 3, figsize=(15, 10))
    #     fig.suptitle('opencv.pyrMeanShiftFiltering + kmeans', fontsize=16)
    #     showRGB(ax[0, 0], image, 'Картинка изначальная')
    #     axes = [ax[0, 1], ax[0, 2], ax[1, 0], ax[1, 1], ax[1, 2]]
    #     for ax, future, sp, sr, maxLevel in zip(axes, futures2, SP, SR, MAX_LEVEL):
    #         showKMeansWithPositions(ax, future.result(),
    #                                 f'kmeans {K}, sp {SP[0]}, sr {SR[0]}, maxLevel {MAX_LEVEL[0]} ')
    #     plt.tight_layout()
    #     plt.show()

    # # 2.4 Выполнение sklearn.cluster.MeanShift по цвету и позиции
    # fig, ax = plt.subplots(2, 3, figsize=(15, 10))
    # fig.suptitle('sklearn.cluster.MeanShift по цвету и позиции', fontsize=16)
    # showRGB(ax[0, 0], image, 'Картинка изначальная')

    # radiuses = [40, 50, 60, 65, 70]
    # axes = [ax[0, 1], ax[0, 2], ax[1, 0], ax[1, 1], ax[1, 2]]
    # data_one_line = data_with_positions.reshape((-1, 5)).astype(np.float32)
    # for ax, radius in zip(axes, radiuses):
    #     ms = MeanShift(bandwidth=radius, bin_seeding=True, n_jobs=-1).fit(data_one_line)

    #     colors = ms.cluster_centers_[:, 2:5].astype(np.uint8)
    #     n = len(colors)
    #     segmented = colors[ms.labels_].reshape(image.shape)
    #     showRGB(ax, segmented, f'{n} clusters, radius={radius}')
    # plt.tight_layout()
    # plt.show()
