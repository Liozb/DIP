import numpy as np
import cv2
import matplotlib.pyplot as plt


def show_spec(img):
    """
    Compute and plot spectrum of image ( amplitude vs phase)
    :param img: Input image
    """
    # Compute the spectrum
    fft = np.fft.fft2(img)
    img_amplitude = np.abs(fft)
    img_phase_spectrum = np.angle(fft)

    # Plot spectrum - amplitude vs phase
    f, arr = plt.subplots(1, 2)

    im1 = arr[0].imshow(np.log(1 + img_amplitude), cmap='gray')
    arr[0].set_title('Amplitude Spectrum')
    plt.colorbar(im1, ax=arr[0])

    im2 = arr[1].imshow(img_phase_spectrum, cmap='gray')
    arr[1].set_title('Phase Spectrum')
    plt.colorbar(im2, ax=arr[1])

    plt.show()


if __name__ == '__main__':
    # Import images
    I = cv2.imread('pic/I.jpg', 0)
    I_n = cv2.imread('pic/I_n.jpg', 0)
    chita = cv2.imread('pic/chita.jpeg', 0)
    zebra = cv2.imread('pic/zebra.jpeg', 0)

    to_run = 'd'

    if to_run == 'a':
        # Show spectrum for image I
        show_spec(I)

        # Show spectrum for image I_n
        show_spec(I_n)

    elif to_run == 'b':
        # Compute spectrum for image I
        I_fft = np.fft.fft2(I)
        I_amplitude = np.abs(I_fft)
        # Compute spectrum for image I_n
        I_n_fft = np.fft.fft2(I_n)
        I_n_amplitude = np.abs(I_n_fft)

        # Compute distance
        amplitude_sub = np.abs(I_amplitude - I_n_amplitude)

        plt.figure()
        plt.imshow(np.log(1 + amplitude_sub), cmap='gray')
        plt.title('Amplitude Subs')
        plt.colorbar()
        plt.show()

    elif to_run == 'c':
        # Compute spectrum for image "chita"
        chita_fft = np.fft.fft2(chita)
        chita_amplitude = np.abs(chita_fft)
        # Compute spectrum for image "zebra"
        zebra_fft = np.fft.fft2(zebra)
        zebra_phase_spectrum = np.angle(zebra_fft)

        # Show spectrum of "chita" and "zebra"
        fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))
        axs[0, 0].imshow(chita, cmap='gray')
        axs[0, 0].set_title('Chita')

        axs[0, 1].imshow(np.log(1 + chita_amplitude), cmap='gray')
        axs[0, 1].set_title('Chita FFT Amplitude Spectrum')

        axs[1, 0].imshow(zebra, cmap='gray')
        axs[1, 0].set_title('Zebra')

        axs[1, 1].imshow(zebra_phase_spectrum, cmap='gray')
        axs[1, 1].set_title('Zebra FFT Phase Spectrum')

        plt.show()

    elif to_run == 'd':
        # Compute spectrum for image "chita"
        chita_fft = np.fft.fft2(chita)
        chita_amplitude = np.abs(chita_fft)

        # Compute spectrum for image "zebra"
        zebra_fft = np.fft.fft2(zebra)
        zebra_phase_spectrum = np.angle(zebra_fft)

        # Use "chita" amplitude and "zebra" phase to make a new image
        chita_amplitude = np.resize(chita_amplitude, zebra_phase_spectrum.shape)
        chita_zebra_fft = chita_amplitude * np.exp(1j * zebra_phase_spectrum)
        chita_zebra_image = np.real(np.fft.ifft2(chita_zebra_fft))

        # Plot result
        plt.subplot(131), plt.imshow(chita, cmap='gray'), plt.title('Chita')
        plt.subplot(132), plt.imshow(zebra, cmap='gray'), plt.title('Zebra')
        plt.subplot(133), plt.imshow(chita_zebra_image, cmap='gray'), plt.title('Chita+Zebra')
        plt.show()
