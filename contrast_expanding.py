import numpy as np
import cv2                      # we used cv2 to import and export images
import matplotlib.pyplot as plt
from pic import *


def show_histogram(img, subplot=True):
    img_flat = img.flatten()
    """
    # first way
    plt.hist(img_flat, bins=256, range=(0, 255))
    plt.show()
    """
    # second way
    hist, bins = np.histogram(img_flat, bins=256, range=(0, 255))
    pix = np.arange(256)
    if subplot:
        return pix, hist
    else:
        plt.bar(pix, hist)
        plt.show()


def contrast_stretching(img):

    new_img = np.copy(img)
    f_max = np.max(img)
    f_min = np.min(img)
    x, y = img.shape
    for i in range(x):
        for j in range(y):
            new_img[i, j] = 255 * (new_img[i, j] - f_min)/(f_max - f_min)

    return new_img


def histogram_equalization(img):
    new_img = np.copy(img)
    h, w = img.shape
    img_flat = img.flatten()
    hist, bins = np.histogram(img_flat, bins=256, range=(0, 255))
    prob = hist/(w*h)
    accumulate = 0
    cdf = np.zeros(256)
    for idx, j in enumerate(prob):
        accumulate += j
        s = np.round(255 * accumulate)
        cdf[idx] = s

    new_img = cdf[img_flat].reshape((h, w))
    return new_img.astype(np.uint8)


if __name__ == '__main__':
    img = cv2.imread('leafs.jpg', 0)

    stretch_img = contrast_stretching(img)
    he_img = histogram_equalization(img)

    # plot images with appropriate contrast
    f, arr = plt.subplots(2, 3)
    arr[0][0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    arr[0][1].imshow(cv2.cvtColor(stretch_img, cv2.COLOR_BGR2RGB))
    arr[0][2].imshow(cv2.cvtColor(he_img, cv2.COLOR_BGR2RGB))

    pix_img, hist_img = show_histogram(img)
    pix_stretch, hist_stretch = show_histogram(stretch_img)
    pix_he, hist_he = show_histogram(he_img)

    arr[1][0].bar(pix_img, hist_img)
    arr[1][1].bar(pix_stretch, hist_stretch)
    arr[1][2].bar(pix_he, hist_he)

    arr[0][0].set_title('original image')
    arr[0][1].set_title('contrast stretching')
    arr[0][2].set_title('histogram equalization')

    plt.show()
