import os

from C_FeatureExtraction.medial_axis import MedialAxis
from C_FeatureExtraction.feature_extractions_constants import *
from C_FeatureExtraction.medial_axis_len import Curve


class ChromosomeLengthComputer:
    def __init__(self, img_path_to_compute, ch_outs_dir_name):
        """

        :param img_path_to_compute: path to chromosome image
        :param ch_outs_dir_name: path to karyo_image_root_dir\feature_extraction\current_chromosome
        """
        self.__chromosome_outs_root_dir = ch_outs_dir_name
        self.__image_path = img_path_to_compute
        self.__medial_axis_computer = None
        self.__medial_axis_curve_len_computer = None
        self.__helper_dir = os.path.join(self.__chromosome_outs_root_dir, CHROMOSOME_LEN_HELPER_DIR)

    def get_chromosome_length(self):
        self.__get_or_create_outs_dir()
        self.__medial_axis_computer = MedialAxis(self.__image_path, self.__helper_dir)
        curve_path = self.__medial_axis_computer.get_medial_axis_img()
        if not Curve.is_valid_curve(curve_path):
            curve_path = self.__medial_axis_computer.get_bin_inv_img_path()
        self.__medial_axis_curve_len_computer = Curve(curve_path, grade=CURVE_POLYNOMIAL_FUNCTION_GRADE)
        chromosome_len = self.__medial_axis_curve_len_computer.get_curve_length()
        return chromosome_len

    def __get_or_create_outs_dir(self):
        if not os.path.exists(self.__helper_dir):
            os.makedirs(self.__helper_dir)
