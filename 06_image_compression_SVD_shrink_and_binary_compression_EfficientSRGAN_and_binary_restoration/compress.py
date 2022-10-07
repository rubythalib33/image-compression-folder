import enum
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


def compressSVD(img, r=100):
    blue, green, red = cv2.split(img)
    
    B_U, B_S, B_VT = np.linalg.svd(blue, full_matrices=False)
    B_S = np.diag(B_S)

    G_U, G_S, G_VT = np.linalg.svd(green, full_matrices=False)
    G_S = np.diag(G_S)

    R_U, R_S, R_VT = np.linalg.svd(red, full_matrices=False)
    R_S = np.diag(R_S)

    U = np.dstack([B_U[:,:r],G_U[:,:r],R_U[:,:r]])
    S = np.dstack([B_S[:r,:r],G_S[:r,:r],R_S[:r,:r]])
    VT = np.dstack([B_VT[:r,:],G_VT[:r,:],R_VT[:r,:]])

    return U, S, VT
    

if __name__ == '__main__':
    import cv2

    path = "asset/0882x4.png"
    img = cv2.imread(path)

    compressed = compressSVD(img)
    # print(compressed)
    print(compressed[0].shape)

    save_compression(compressed, "asset/result.pc")
    cv2.imwrite("asset/0882x4.jpg", img)