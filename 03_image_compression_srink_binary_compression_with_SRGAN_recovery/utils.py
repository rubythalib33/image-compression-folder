from math import log10, sqrt
import numpy as np


def psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if(mse == 0):  # MSE is zero means no noise is present in the signal .
                  # Therefore PSNR have no importance.
        return 100
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr



def bit8to4(img):
    return (img//16).astype(np.uint8)


def bit4to8(img):
    return img*16