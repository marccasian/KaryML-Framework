import cv2
import os
import numpy as np

from C_FeatureExtraction.feature_extractions_constants import *


class MedialAxis:
    def __init__(self, img_path, outputs_dir):
        self.__img_path = img_path
        self.__img = None
        self.__outputs_root_dir = outputs_dir
        self.__img_name = os.path.basename(self.__img_path)
        self.__img_outputs_dir = os.path.join(self.__outputs_root_dir, self.__img_name.split(".")[0])
        self.__img_outputs_feature_extractor_dir = os.path.join(self.__img_outputs_dir,
                                                                FEATURE_EXTRACTION_IMGS_ROOT_DIR)
        self.__img_outputs_feature_extractor_medial_axis_dir = os.path.join(self.__img_outputs_feature_extractor_dir,
                                                                            MEDIAL_AXIS_IMAGES_DIR)
        self.__medial_axis_img = None
        self.__medial_axis_img_path = os.path.join(self.__img_outputs_feature_extractor_medial_axis_dir,
                                                   "%s-final.bmp" % self.__img_name.split(".")[0])

        self.__bin_inv_image_path = os.path.join(
            self.__img_outputs_feature_extractor_medial_axis_dir,
            "%s_bin_inv.bmp" % self.__img_name.split(".")[0])

        self.__prepare_feature_extractor_dirs()

    def get_bin_inv_img_path(self):
        if not os.path.exists(self.__bin_inv_image_path):
            self.__compute_medial_axis_img()
        return self.__bin_inv_image_path

    def __prepare_feature_extractor_dirs(self):
        if not os.path.exists(self.__img_outputs_feature_extractor_medial_axis_dir):
            os.makedirs(self.__img_outputs_feature_extractor_medial_axis_dir)

    def get_medial_axis_img(self):
        if self.__medial_axis_img is None:
            self.__compute_medial_axis_img()
        return self.__medial_axis_img_path

    def set_img_object(self, img):
        self.__img = img

    @staticmethod
    def __border_image(img, border_val):
        new_img = np.ones(img.shape + np.ones_like(img.shape) * 2) * border_val
        new_img[1:-1, 1:-1] = img
        return new_img

    def __compute_medial_axis_img(self):
        if self.__img is None:
            self.__img = cv2.imread(self.__img_path, 0)
        self.__img = self.__border_image(self.__img, 255)

        size = np.size(self.__img)
        self.__medial_axis_img = np.zeros(self.__img.shape, np.float64)

        ret, img = cv2.threshold(self.__img, 245, 255, cv2.THRESH_BINARY_INV)
        cv2.imwrite(self.__bin_inv_image_path, img[1:-1, 1:-1])
        element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        done = False
        doesnt_modiff = MEDIAL_AXIS_ITERATION_THREASHOLD
        while not done:
            if doesnt_modiff == 0:
                break
            eroded = cv2.erode(img, element)
            temp = cv2.dilate(eroded, element)
            temp = cv2.subtract(img, temp)
            self.__medial_axis_img = cv2.bitwise_or(self.__medial_axis_img, temp)
            if (eroded == img).all():
                doesnt_modiff -= 1
            else:
                doesnt_modiff = MEDIAL_AXIS_ITERATION_THREASHOLD
            img = eroded.copy()

            zeros = size - cv2.countNonZero(img)
            if zeros == size:
                done = True
        cv2.imwrite(self.__medial_axis_img_path, self.__medial_axis_img)

        for i in range(self.__medial_axis_img.shape[0]):
            for j in range(self.__medial_axis_img.shape[1]):
                if self.__medial_axis_img[i][j] > 0:
                    self.__img[i][j] = 255
        cv2.imwrite(os.path.join(
            self.__img_outputs_feature_extractor_medial_axis_dir,
            "%s-medial.bmp" % self.__img_name.split(".")[0]),
            self.__img[1:-1, 1:-1])
