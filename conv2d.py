import numpy as np
import cv2
import matplotlib.pyplot as plt


def conv2d(img, kernel):
    """
    The function implements a convolution of a kernel on the img
    :param img: The image we want to do a convolution on
    :param kernel: The kernel used for the convolution
    :return: the result of the convolution
    """
    kernel_size = kernel.shape[0]
    pad_length = int(kernel_size/2 - 0.5)    # we assume kernel shape is odd
    result = np.zeros(img.shape, dtype=np.uint8)

    # rotate the kernel
    flip_kernel = np.flip(kernel, 0)
    flip_kernel = np.flip(flip_kernel, 1)
    img_pad = np.pad(img, pad_length, constant_values=0)
    x, y = img_pad.shape
    for i in range(x-kernel_size + 1):
        for j in range(y-kernel_size + 1):
            mat = img_pad[i:i+kernel_size, j:j+kernel_size]
            conv_result = np.sum(np.multiply(mat, flip_kernel))
            result[i, j] = np.clip(conv_result, 0, 255)
    return result


def conv2d_check():
    img = np.matrix([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
    kernel = np.matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    print(conv2d(img, kernel))


if __name__ == '__main__':
    # conv2d_check()
    img = cv2.imread('zebra.jpeg', 0)
    kernel = np.matrix([[1, 1, 1], [1, 1, 1], [1, 1, 1]]) * 1/9

    # By implementing a box filter on the image we expect to get blurred image
    new_img = conv2d(img, kernel)

    cv2.imwrite('box kernel.jpeg', new_img)
    f, arr = plt.subplots(1, 2)
    arr[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    arr[1].imshow(cv2.cvtColor(new_img, cv2.COLOR_BGR2RGB))
    arr[0].set_title('original image')
    arr[1].set_title('image after box filter')
    plt.show()
