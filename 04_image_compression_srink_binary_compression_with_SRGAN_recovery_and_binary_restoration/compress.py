from utils import *
import numpy as np
from PIL import Image
from albumentations import Resize


def compress_shrink(img: np.ndarray, ratio:int):
    resized = Resize(width=img.shape[1]//ratio, height=img.shape[0]//ratio, interpolation=Image.BICUBIC)(image=img)["image"]
    return resized


def combinebit(img_4_bit):
    h,w, _ = img_4_bit.shape

    if h % 2 == 0:
        top_img = img_4_bit[:h//2]
        bottom_img = img_4_bit[h//2:]

    else:
        raise ValueError("height shape should be even, please resize your height of image into become even number")

    return top_img+(bottom_img*16)


def compress_binary(img):
    img_4_bit = bit8to4(img)
    img_combined = combinebit(img_4_bit)

    return img_combined


def compress(img, ratio=4):
    shrink = compress_shrink(img, ratio=ratio)
    binary = compress_binary(shrink)

    return binary

if __name__ == '__main__':
    import cv2

    path = "asset/HR_images/0810x4.png"
    img = cv2.imread(path)

    compressed = compress(img)

    cv2.imwrite("asset/Compressed/result2.png", compressed)