import cv2
import numpy as np
import os
import sys


def Diezmado(D, image_gray):
    #assert (D > 1 and type(D) is int, "D debe ser mayor a 1 y entero")
    image_gray_fft = np.fft.fft2(image_gray)
    image_gray_fft_shift = np.fft.fftshift(image_gray_fft)

    # fft visualization
    image_gray_fft_mag = np.absolute(image_gray_fft_shift)
    image_fft_view = np.log(image_gray_fft_mag + 1)
    image_fft_view = image_fft_view / np.max(image_fft_view)

    # pre-computations
    num_rows, num_cols = (image_gray.shape[0], image_gray.shape[1])
    enum_rows = np.linspace(0, num_rows - 1, num_rows)
    enum_cols = np.linspace(0, num_cols - 1, num_cols)
    col_iter, row_iter = np.meshgrid(enum_cols, enum_rows)
    half_size_r = num_rows / 2 - 1  # here we assume num_rows = num_columns
    half_size_c = num_cols / 2 - 1  # here we assume num_rows = num_columns

    # low pass filter mask
    low_pass_mask = np.zeros_like(image_gray)
    freq_cut_off = 1 / D  # it should less than 1
    radius_cut_off_r = int(freq_cut_off * half_size_r)
    radius_cut_off_c = int(freq_cut_off * half_size_c)
    idx_lp = ((((row_iter - half_size_r) ** 2)/(radius_cut_off_r**2))+(((col_iter - half_size_c) ** 2)/
                                                                           (radius_cut_off_c**2))) < 1
    low_pass_mask[idx_lp] = 1

    # filtering via FFT
    mask = low_pass_mask  # can also use high or band pass mask
    fft_filtered = image_gray_fft_shift * mask
    image_filtered = np.fft.ifft2(np.fft.fftshift(fft_filtered))
    image_filtered = np.absolute(image_filtered)
    image_filtered /= np.max(image_filtered)

    # Decimation
    image_decimated = image_filtered[::D, ::D]
    return image_decimated



def Interpolacion(I, image_gray):
    # Interpolation
    # insert zeros
    rows, cols = image_gray.shape
    num_of_zeros = I
    image_zeros = np.zeros((num_of_zeros * rows, num_of_zeros * cols), dtype=image_gray.dtype)
    image_zeros[::num_of_zeros, ::num_of_zeros] = image_gray


    ## Filtrado FFT
    image_gray_fft = np.fft.fft2(image_zeros)
    image_gray_fft_shift = np.fft.fftshift(image_gray_fft)

    # fft visualization
    image_gray_fft_mag = np.absolute(image_gray_fft_shift)
    image_fft_view = np.log(image_gray_fft_mag + 1)
    image_fft_view = image_fft_view / np.max(image_fft_view)

    # pre-computations
    num_rows, num_cols = (image_zeros.shape[0], image_zeros.shape[1])
    enum_rows = np.linspace(0, num_rows - 1, num_rows)
    enum_cols = np.linspace(0, num_cols - 1, num_cols)
    col_iter, row_iter = np.meshgrid(enum_cols, enum_rows)
    half_size_r = num_rows / 2 - 1  # here we assume num_rows = num_columns
    half_size_c = num_cols / 2 - 1  # here we assume num_rows = num_columns

    # low pass filter mask
    low_pass_mask = np.zeros_like(image_zeros)
    freq_cut_off = 1 / I  # it should less than 1
    radius_cut_off_r = int(freq_cut_off * half_size_r)
    radius_cut_off_c = int(freq_cut_off * half_size_c)
    idx_lp = ((((row_iter - half_size_r) ** 2) / (radius_cut_off_r ** 2)) + (((col_iter - half_size_c) ** 2) /
                                                                             (radius_cut_off_c ** 2))) < 1
    low_pass_mask[idx_lp] = 1

    # filtering via FFT
    mask = low_pass_mask  # can also use high or band pass mask
    fft_filtered = image_gray_fft_shift * mask
    image_filtered = np.fft.ifft2(np.fft.fftshift(fft_filtered))
    image_filtered = np.absolute(image_filtered)
    image_filtered /= np.max(image_filtered)

    return image_filtered



def descom(image,N):
    H = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    V = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    D = np.array([[2, -1, -2], [-1, 4, -1], [-2, -1, 2]])
    L = np.array([[1 / 16, 1 / 8, 1 / 16], [1 / 8, 1 / 4, 1 / 8], [1 / 16, 1 / 8, 1 / 16]])
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lista_in = [image_gray]
    lista_out = []

    for i in range (N):
        H_convolved = cv2.filter2D(lista_in[i], -1, H)
        V_convolved = cv2.filter2D(lista_in[i], -1, V)
        D_convolved = cv2.filter2D(lista_in[i], -1, D)
        L_convolved = cv2.filter2D(lista_in[i], -1, L)

        IH = Diezmado(2, H_convolved)
        IV = Diezmado(2, V_convolved)
        ID = Diezmado(2, D_convolved)
        IL = Diezmado(2, L_convolved)

        lista_in.append(IL)
        if i<N-1:
         lista_out.append([IH,IV,ID])
        else:
         lista_out.append([IH, IV,ID,IL])

    return lista_out
