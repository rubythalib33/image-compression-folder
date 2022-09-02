from utils import *


def split(img):
    bottom_img = bit8to4(img)
    top_img = img - bottom_img*16
    return np.vstack((top_img, bottom_img))


def recovery_binary(img):
    split_img = split(img)
    img_8_bit = bit4to8(split_img)

    return img_8_bit