import numpy as np
import cv2                      # we used cv2 only to import and export images
import matplotlib.pyplot as plt
from pic import *


def interpolation_bi_linear(img):
    # symmetric bi-linear interpolation
    factor = 2
    i = 1
    X = img.shape[0]
    Y = img.shape[1]
    pic_pad = np.pad(img, 1, mode='constant')                                     # pad for edges
    pic_after_inter = np.zeros((X*factor, Y*factor), dtype=np.uint8)              # we expand resolution by factor

    for x in range(X*factor):                                                     # go through the matrix rows
        for y in range(Y*factor):                                                 # go through the matrix columns
            if (x + y) % factor == 0:                                             # original pic values moved by factor
                pic_after_inter[x][y] = img[x//factor][y//factor]
                i = 1
            else:
                p1 = i / factor * pic_pad[x // factor + 1][y // factor + 2] + (1 - i / factor) * \
                     pic_pad[x // factor + 1][y // factor + 1]
                p2 = i / factor * pic_pad[x // factor + 2][y // factor + 2] + (1 - i / factor) * \
                     pic_pad[x // factor + 2][y // factor + 1]

                pic_after_inter[x][y] = int(round(0.5 * (p1+p2)))
                i += 1
    return pic_after_inter


def create_bicubic_matrix():
    cube = 16
    idx = 0
    mat = np.zeros((cube, cube))
    for i in range(4):
        for j in range(4):
            X_values = np.matrix([-2, -1, 0, 1])
            Y_values = np.matrix([-2, -1, 0, 1])
            X_values = np.power(X_values, i)
            Y_values = np.power(Y_values, j)
            multiply = X_values.transpose() @ Y_values
            multiply = np.reshape(multiply, (cube, 1))
            mat[:, idx] = np.reshape(multiply, (16,))
            idx = idx + 1
    return mat


def bicubic_interpolation(img, factor=2):
    # Bicubic interpolation
    matrix = create_bicubic_matrix()
    mat_inv = np.linalg.inv(matrix)

    X = img.shape[0]
    Y = img.shape[1]

    # Pad for edges of the image
    img_pad = np.pad(img, ((1, 2), (1, 2)), mode='reflect')
    img_inter = np.zeros((X * factor, Y * factor), dtype=np.uint8)

    # Go over the image pixels an compute interpolation
    for i in range(X * factor):
        for j in range(Y * factor):
            x = i / factor
            y = j / factor
            x1 = int(np.floor(x))
            y1 = int(np.floor(y))
            neighbors = img_pad[x1:x1 + 4, y1:y1 + 4]
            x -= x1
            y -= y1
            cubic_x = np.matrix([1, x, x ** 2, x ** 3])
            cubic_y = np.matrix([1, y, y ** 2, y ** 3]).transpose()
            coeffs = np.reshape(mat_inv @ neighbors.flatten(), (4, 4))
            result = np.sum(cubic_x * (coeffs * cubic_y))
            img_inter[i, j] = np.clip(result, 0, 255)

    return img_inter


if __name__ == '__main__':
    peppers = cv2.imread('peppers.jpg', 0)
    bi_linear_img = interpolation_bi_linear(peppers)
    bicubic_img = bicubic_interpolation(peppers)

    # show results
    f, arr = plt.subplots(1, 3)
    arr[0].imshow(cv2.cvtColor(peppers, cv2.COLOR_BGR2RGB))
    arr[1].imshow(cv2.cvtColor(bi_linear_img, cv2.COLOR_BGR2RGB))
    arr[2].imshow(cv2.cvtColor(bicubic_img, cv2.COLOR_BGR2RGB))
    arr[0].set_title('original image')
    arr[1].set_title('bi-linear interpolation')
    arr[2].set_title('bicubic interpolation')
    plt.show()

