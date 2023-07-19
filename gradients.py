import numpy as np
import cv2
import matplotlib.pyplot as plt
from conv2d import conv2d


def diff_filter(img):
    """
    :param img: accept image as input
    :return: The gradient filter in the horizontal direction of the image
    """
    diff_fil = np.array([-1, 0, 1])
    filter_length = len(diff_fil)
    img_pad = np.pad(img, 1, constant_values=0)
    result = np.zeros(img.shape, dtype=np.uint8)

    x, y = img.shape

    for i in range(x):
        for j in range(y-filter_length + 1):
            mat = img_pad[i, j:j + filter_length]
            conv_result = np.sum(np.multiply(mat, diff_fil))
            result[i, j] = conv_result

    return result


def compute_mse(img, new_img):
    x, y = img.shape
    sub = img - new_img
    result = np.sqrt((1/(x*y) * np.sum(sub)))
    return result


if __name__ == '__main__':
    I = cv2.imread('pic/I.jpg', 0)
    I_n = cv2.imread('pic/I_n.jpg', 0)

    # Implement a gradient filter in the horizontal direction on both I, and I_n pictures
    I_diff = diff_filter(I)
    I_n_diff = diff_filter(I_n)

    # Implement Gaussian filter on noisy image (our case - I_n)
    kernel_size = 7
    sigma = 10

    # Generate the Gaussian kernel
    gaussian_kernel = cv2.getGaussianKernel(kernel_size, sigma)
    gaussian_kernel_2d = np.outer(gaussian_kernel, gaussian_kernel)

    # The gaussian kernel is symmetric, so applying filtering is the same as convolution
    I_dn = conv2d(I_n, gaussian_kernel_2d)
    print("MSE between I to I_dn is :", compute_mse(I, I_dn))

    # Implement Gaussian filter on noisy image after smoothing (our case - I_dn)
    I_dn_diff = diff_filter(I_dn)

    # Implement the Sobel filter in the horizontal direction
    sobel_x = np.matrix([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    I_n_sobel = conv2d(I_n, sobel_x)

    # save images
    cv2.imwrite('I_diff.jpg', I_diff)
    cv2.imwrite('I_n_diff.jpg', I_n_diff)
    cv2.imwrite('I_dn.jpg', I_dn)
    cv2.imwrite('I_dn_diff.jpg', I_dn_diff)
    cv2.imwrite('I_n_sobel.jpg', I_n_sobel)

    # Plot images
    f, arr = plt.subplots(5, 2)
    arr[0][0].imshow(cv2.cvtColor(I, cv2.COLOR_BGR2RGB))
    arr[0][1].imshow(cv2.cvtColor(I_n, cv2.COLOR_BGR2RGB))
    arr[1][0].imshow(cv2.cvtColor(I_diff, cv2.COLOR_BGR2RGB))
    arr[1][1].imshow(cv2.cvtColor(I_n_diff, cv2.COLOR_BGR2RGB))
    arr[2][1].imshow(cv2.cvtColor(I_dn, cv2.COLOR_BGR2RGB))
    arr[3][1].imshow(cv2.cvtColor(I_dn_diff, cv2.COLOR_BGR2RGB))
    arr[4][1].imshow(cv2.cvtColor(I_n_sobel, cv2.COLOR_BGR2RGB))

    arr[0][0].set_title('original image')
    arr[0][1].set_title('original image with noise')
    arr[1][0].set_title('original image - horizontal gradient')
    arr[1][1].set_title('noisy image - horizontal gradient')
    arr[2][1].set_title('noisy image - gaussian filter')
    arr[3][1].set_title('noisy image after gaussian filter - horizontal gradient')
    arr[4][1].set_title('noisy image - sobel filter(in horizontal direction)')

    arr[2][0].set_visible(False)
    arr[3][0].set_visible(False)
    arr[4][0].set_visible(False)

    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.4, hspace=0.4)
    plt.show()
