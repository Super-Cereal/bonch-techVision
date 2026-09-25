import numpy as np


def convolution(image, kernel):
    assert kernel.shape[0] == kernel.shape[1]
    assert kernel.shape[0] % 2 == 1

    center = (kernel.shape[0] - 1) // 2
    output = np.zeros_like(image, dtype=np.float32)

    for i in range(0, image.shape[0]):
        for j in range(0, image.shape[1]):
            acc = 0.0

            for delta_i in range(-center, center + 1):
                ni = i + delta_i

                if ni < 0 or ni >= image.shape[0]:
                    continue

                for delta_j in range(-center, center + 1):
                    nj = j + delta_j

                    if nj < 0 or nj >= image.shape[1]:
                        continue

                    multiplier = kernel[delta_i + center, delta_j + center]

                    acc += image[ni, nj] * multiplier

            output[i, j] = np.clip(acc, 0, 255)

    return output
