import cv2
import numpy as np
import os
from utils import *

def recovery(usvt):
    U, S, VT = usvt
    B_U, G_U, R_U = np.dsplit(U, 3)
    B_S, G_S, R_S = np.dsplit(S,3 )
    B_VT, G_VT, R_VT = np.dsplit(VT,3)
    
    blue = B_VT.squeeze() @ B_S.squeeze() @ B_U.squeeze()
    green = G_VT.squeeze() @ G_S.squeeze() @ G_U.squeeze()
    red = R_VT.squeeze() @ R_S.squeeze() @ R_U.squeeze()

    return np.dstack([blue,green,red]).astype(np.uint8)


if __name__ == '__main__':
    compressed_path = 'asset/result.pc'

    compressed = load_compression(compressed_path)
    recovered = recovery(compressed)
    print(recovered.shape)
    recovery_path = 'asset/rec.jpg'
    cv2.imwrite(recovery_path, recovered)