import os

from cv2 import cv2
from A_Segmentation.constants import *


def read_image(image_path, inverted = False):
    if inverted:
        print("Inverted")
        return cv2.bitwise_not(cv2.imread(image_path))
    a = cv2.imread(image_path)
    if a is None:
        msg = "Couldn't load image from %s" % image_path
        raise Exception(msg)
    return a


def write_image(image, file_path):
    cv2.imwrite(image, file_path)


def eq_pixels(px1, px2):
    return px1[0] == px2[0] and px1[1] == px2[1] and px1[2] == px2[2]


def gt_pixels(px1, px2):
    return px1[0] > px2[0] and px1[1] > px2[1] and px1[2] > px2[2]


def __mod_lt(x1, x2, m):
    return abs(x1 - x2) < m


def almost_eq_pixels(px1, px2, val=ALMOST_EQ_VALUE):
    return __mod_lt(px1[0], px2[0], val) \
           and __mod_lt(px1[1], px2[1], val) \
           and __mod_lt(px1[2], px2[2], val)


def lt_pixels(px1, px2):
    return px1[0] < px2[0] and px1[1] < px2[1] and px1[2] < px2[2]


def get_all_images(dir_path="imgs"):
    return [os.path.join(dir_path, f)
            for f in os.listdir(dir_path)
            if f.lower().endswith('.jpg')
            or f.lower().endswith('.bmp')
            or f.lower().endswith('.jpeg')
            ]
