import math
import os

import imutils
from cv2 import cv2
from matplotlib.pyplot import axhline

from B_Straightening import compute_projection_vector
from B_Straightening import rotate_image
from B_Straightening.image_trimmer import TrimImage
from C_FeatureExtraction.chromosome_length_computer import ChromosomeLengthComputer
from C_FeatureExtraction.feature_extractions_constants import *


class CentromereDetector:
    def __init__(self, chromosome_img):
        self.image_path = chromosome_img
        self.current_angle = 0
        self.best_angle = 0
        self.best_score = math.inf
        self.best_image_path = None
        self.best_horizontal_projection_vector = None
        self.p1 = None  # highest point in horizontal projection vector
        self.p2 = None  # second highest point in horizontal projection vector
        self.p3 = None  # min value between p1 and p2 in horizontal projection vector
        self.w1 = 0.5  # tuning parameters to control the weight of each term in the rotation score S
        self.w2 = 0.5  # where w1 < 1, w2 < 1 and w1 + w2 = 1
        self.best_min_position = None

    def compute_centromere_position(self):
        self.__compute_centromere_position()
        return self.best_min_position, self.best_image_path

    def __compute_centromere_position(self):
        for self.current_angle in range(0, 180, 5):
            rotate_image_obj = rotate_image.RotateImage()
            rotate_image_obj.rotate_image(self.image_path, self.current_angle)
            rotated_image_path = rotate_image_obj.get_output_image_path()
            compute_projection_vector_obj = compute_projection_vector.ComputeProjectionVector()
            h_vector = compute_projection_vector_obj.get_horizontal_projection_vector(rotated_image_path)
            score, min_position = self.__compute_horizontal_vector_score(h_vector)
            if score < self.best_score:
                self.best_score = score
                self.best_angle = self.current_angle
                self.best_min_position = min_position
                self.best_image_path = rotated_image_path

    @staticmethod
    def __compute_score(p1, p2, p3, w1, w2):
        r1 = abs(p1 - p2) / (p1 + p2)
        r2 = p3 / (p1 + p2)
        return w1 * r1 + w2 * r2

    def __compute_horizontal_vector_score(self, h_vector):
        score = math.inf
        min_position = None
        max_position = h_vector.index(max(h_vector))
        max_value = max(h_vector)
        for i in range(len(h_vector)):
            if i != max_position:
                start = min(i, max_position)
                end = max(i, max_position)
                if end - start > 1:
                    min_value = min(h_vector[start + 1:end])
                    current_min_position = h_vector[start + 1:end].index(min_value)
                    current_score = self.__compute_score(max_value, h_vector[i], min_value, self.w1, self.w2)
                    if current_score < score:
                        score = current_score
                        min_position = current_min_position + start + 1
                else:
                    continue
        return score, min_position


class ShortChromatidRation:
    def __init__(self, chromosome_img, outs_dir):
        """

        :param chromosome_img: path to chromosome image
        :param outs_dir: chromosome feature extraction  dir
        """
        self.__img_path = chromosome_img
        self._centromere_detector = CentromereDetector(self.__img_path)
        self.img = cv2.imread(self.__img_path)
        self.__helper_dir = os.path.join(outs_dir, SHORT_CHROMATID_RATIO_HELPER_DIR)
        self.__first_chromatid_image_path = os.path.join(self.__helper_dir, FIRST_CHROMATID_FILE_NAME)
        self.__second_chromatid_image_path = os.path.join(self.__helper_dir, SECOND_CHROMATID_FILE_NAME)
        self.__centromere_image_path = os.path.join(self.__helper_dir, CHROMOSOME_HIGHLIGHTED_CENTROMERE_FILE_NAME)
        self.__best_img = None
        self.__image_trimmer = TrimImage()
        self.__centromere_position, self.best_img_path = None, None
        self.__short_chromatid_ratio = None
        self.__centromere_relative_position = None

    def get_centromere_relative_position(self):
        if self.__centromere_relative_position is None:
            self.__centromere_position, self.best_img_path = self._centromere_detector.compute_centromere_position()
        return self.__centromere_relative_position

    def get_short_chromatid_ratio(self):
        if self.__short_chromatid_ratio is None:
            self.__compute_short_chromatid_ratio()
        return self.__short_chromatid_ratio

    def __compute_short_chromatid_ratio(self):
        if not os.path.exists(self.__helper_dir):
            os.makedirs(self.__helper_dir)
        self.__centromere_position, self.best_img_path = self._centromere_detector.compute_centromere_position()
        self.__best_img = cv2.imread(self.best_img_path)
        img1 = self.__best_img[:self.__centromere_position, :]
        cv2.imwrite(self.__first_chromatid_image_path, img1)
        self.__image_trimmer.trim(self.__first_chromatid_image_path)
        img2 = self.__best_img[self.__centromere_position:, :]
        cv2.imwrite(self.__second_chromatid_image_path, img2)
        self.__image_trimmer.trim(self.__second_chromatid_image_path)
        self.__best_img[self.__centromere_position, :] = (255, 0, 0)
        cv2.imwrite(self.__centromere_image_path, self.__best_img)
        first_chromatid_length = ChromosomeLengthComputer(self.__first_chromatid_image_path,
                                                          self.__helper_dir).get_chromosome_length()
        second_chromatid_length = ChromosomeLengthComputer(self.__second_chromatid_image_path,
                                                           self.__helper_dir).get_chromosome_length()
        ch_length = ChromosomeLengthComputer(self.best_img_path, self.__helper_dir).get_chromosome_length()
        short_ch_len = min(first_chromatid_length, second_chromatid_length)
        # short chromatid len correspond to centromere line
        self.__centromere_relative_position = short_ch_len
        long_ch_len = max(first_chromatid_length, second_chromatid_length)
        print("short_ch = " + str(short_ch_len))
        print("long_ch = " + str(long_ch_len))
        print("ch_len = " + str(ch_length))
        self.__short_chromatid_ratio = short_ch_len / long_ch_len


def test_1(obj):
    import matplotlib.pyplot as plt

    a = 0
    img_bg = obj.img
    # img_bg = imutils.rotate_bound(img_bg, 90)[:, 1:, :]
    # img_bg = imutils.rotate_bound(obj.orig_threshold_img, 90)
    plt.imshow(img_bg, zorder=0)
    centromere_position = obj.get_centromere_relative_position()
    plt.axhline(linewidth=4, color='r', y=int(centromere_position))

    plt.show()


if __name__ == "__main__":
    img_path = r'__disertation_experiments\dataset\1\1\contrast_split\straight\13-0.bmp'
    out_dir = r'__disertation_experiments\dataset\1\1\Outputs\13-0'
    obj = ShortChromatidRation(img_path, out_dir)
    print(obj.get_short_chromatid_ratio())
    test_1(obj)
