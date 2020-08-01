import json
import os
import math
import pickle

from a_Common.my_logger import LOGGER
from A_Segmentation.constants import *
import A_Segmentation.common_operations as common_operations
import B_Straightening.compute_projection_vector as compute_projection_vector
from C_FeatureExtraction.centromere_related_features import ShortChromatidRation
from C_FeatureExtraction.bandage_profile_feature import BandageFeature
from C_FeatureExtraction.feature_extractions_constants import *
import C_FeatureExtraction.medial_axis as medial_axis
import C_FeatureExtraction.medial_axis_len as medial_axis_len


class FeatureExtractor:
    FEATURE_USED_FOR_CLUSTERING = list()
    FEATURE_USED_FOR_CLUSTERING_WEIGHTS = list()
    CHROMOSOME_LEN_FEATURE_FLAG = 0x01
    CHROMOSOME_SHORT_CHROMATID_RATIO_FEATURE_FLAG = 0x02
    CHROMOSOME_BANDAGE_PROFILE_FEATURE_FLAG = 0x04
    CHROMOSOME_CHROMOSOME_AREA_FEATURE_FLAG = 0x08

    def __init__(self, chromosomes_dir, root_dir, outs_dir, straighten=False, weights=None):
        self.features = dict()
        self.w1 = 0.5
        self.w2 = 0.5
        self.logger = LOGGER.getChild("FeatureExtractor")
        self.chromosomes_dir = chromosomes_dir
        self.root_dir = root_dir
        self.__segments_file = os.path.join(self.root_dir, SEGMENTS_FILE_NAME)
        self.features_out_file = chromosomes_dir + "_features.txt"
        self.features_out_json_file = chromosomes_dir + "_features.json"
        self.features_out_pickle_file = chromosomes_dir + "_features.pickle"
        self.features_weights_out_file = chromosomes_dir + "_features_weights.txt"
        self.__outputs_root_dir = outs_dir
        self.__straighten = straighten
        self.__nr_of_bandages_to_extract = 0
        self.weights = weights

    def get_all_images(self):
        return [os.path.join(self.chromosomes_dir, f)
                for f in os.listdir(self.chromosomes_dir)
                if f.lower().endswith('.bmp')
                ]

    def extract_features(self, features=0x0f):
        self.__init_feature_dict()
        features_no = sum([1 for _ in [1 for i in [self.CHROMOSOME_LEN_FEATURE_FLAG,
                                                   self.CHROMOSOME_SHORT_CHROMATID_RATIO_FEATURE_FLAG,
                                                   self.CHROMOSOME_BANDAGE_PROFILE_FEATURE_FLAG,
                                                   self.CHROMOSOME_CHROMOSOME_AREA_FEATURE_FLAG]
                                       if features & i != 0]])
        if features & self.CHROMOSOME_LEN_FEATURE_FLAG != 0:
            if self.weights is not None:
                self.FEATURE_USED_FOR_CLUSTERING.append([CHROMOSOME_LEN_KEY, self.weights[1]])
            else:
                self.FEATURE_USED_FOR_CLUSTERING.append([CHROMOSOME_LEN_KEY, 1 / features_no])

            self.__extract_len_feature()
        if features & self.CHROMOSOME_SHORT_CHROMATID_RATIO_FEATURE_FLAG != 0:
            if self.weights is not None:
                self.FEATURE_USED_FOR_CLUSTERING.append([SHORT_CHROMATID_RATIO_KEY, self.weights[2]])
            else:
                self.FEATURE_USED_FOR_CLUSTERING.append([SHORT_CHROMATID_RATIO_KEY, 1 / features_no])

            self.__extract_short_chromatid_ratio_feature()
        if features & self.CHROMOSOME_BANDAGE_PROFILE_FEATURE_FLAG != 0:
            if self.weights is not None:
                self.FEATURE_USED_FOR_CLUSTERING.append([BANDAGE_PROFILE_KEY, self.weights[3]])
            else:
                self.FEATURE_USED_FOR_CLUSTERING.append([BANDAGE_PROFILE_KEY, 1 / features_no])

            self.__extract_banding_profile()
        if features & self.CHROMOSOME_CHROMOSOME_AREA_FEATURE_FLAG != 0:
            if self.weights is not None:
                self.FEATURE_USED_FOR_CLUSTERING.append([CHROMOSOME_AREA_KEY, self.weights[4]])
            else:
                self.FEATURE_USED_FOR_CLUSTERING.append([CHROMOSOME_AREA_KEY, 1 / features_no])
            self.__extract_area_feature()
        self.dump_in_file_features()

    def __init_feature_dict(self):
        segments = list()

        with open(self.__segments_file, "rb") as fp:
            segments = pickle.load(fp)
        for file in self.get_all_images():
            self.features[file] = dict()
            self.features[file][CHROMOSOME_NUMBER_KEY] = ".".join(os.path.basename(file).split(".")[:-1]).split("-")[
                0].strip("_c")
            self.features[file][CHROMOSOME_COLOR_KEY] = segments[int(self.features[file][CHROMOSOME_NUMBER_KEY])].color

    def __extract_len_feature(self):
        self.logger.debug("Start len feature extraction")
        for file in self.get_all_images():
            self.logger.debug("Extract len feature for %s" % os.path.basename(file))
            self.features[file][CHROMOSOME_LEN_KEY] = float(self.__get_chromosome_len(file))

    def __extract_short_chromatid_ratio_feature(self):
        self.logger.debug("Start centromeric index feature extraction")
        for file in self.get_all_images():
            self.logger.debug("Extract centromeric index feature for %s" % os.path.basename(file))
            feature_calculator = ShortChromatidRation(file, os.path.join(
                self.__outputs_root_dir,
                ".".join(os.path.basename(file).split(".")[:-1])))
            self.features[file][SHORT_CHROMATID_RATIO_KEY] = float(feature_calculator.get_short_chromatid_ratio())
            self.features[file][CENTROMERE_POSITION_KEY] = float(feature_calculator.get_centromere_relative_position())
            self.features[file][BEST_IMAGE_PATH_KEY] = feature_calculator.best_img_path

    def extract_short_chromatid_ratio_feature_for_one_image(self, file):
        self.logger.debug("Extract centromeric index feature for %s" % os.path.basename(file))
        feature_calculator = ShortChromatidRation(file, os.path.join(
            self.__outputs_root_dir,
            ".".join(os.path.basename(file).split(".")[:-1])))
        self.features[file] = dict()
        self.features[file][SHORT_CHROMATID_RATIO_KEY] = feature_calculator.get_short_chromatid_ratio()
        self.features[file][CENTROMERE_POSITION_KEY] = feature_calculator.get_centromere_relative_position()
        self.features[file][BEST_IMAGE_PATH_KEY] = feature_calculator.best_img_path

    def __extract_short_chromatid_ratio_feature_old(self):
        for file in self.get_all_images():
            compute_projection_vector_obj = compute_projection_vector.ComputeProjectionVector()
            h_vector = compute_projection_vector_obj.get_horizontal_projection_vector(file)
            min_position = self.__compute_horizontal_vector_score(h_vector)
            chromosome_len = self.features[file][CHROMOSOME_LEN_KEY]
            short_chromatid_len = min(min_position, chromosome_len - min_position)
            self.features[file][SHORT_CHROMATID_RATIO_KEY] = short_chromatid_len / chromosome_len
            self.features[file][CENTROMERE_POSITION_KEY] = min_position

    def __extract_banding_profile(self):
        bandage_profile_objs = list()
        self.__nr_of_bandages_to_extract = math.inf
        for file in self.get_all_images():
            current_obj = BandageFeature(file)
            intersection_points = current_obj.get_intersection_points()
            if len(intersection_points) < self.__nr_of_bandages_to_extract:
                self.__nr_of_bandages_to_extract = len(intersection_points)
            bandage_profile_objs.append(current_obj)
        print("Will extract %d bandages for each chromosome" % self.__nr_of_bandages_to_extract)
        for bandage_profile_obj in bandage_profile_objs:
            bandage_profile = bandage_profile_obj.get_bandage_profile(
                intersection_points_nr=self.__nr_of_bandages_to_extract)
            self.features[bandage_profile_obj.image_path][BANDAGE_PROFILE_KEY] = bandage_profile

    def __get_chromosome_len(self, file):
        """

        :param file: straighten image file
        :return: chromosome length
        """
        if self.__straighten:
            img = common_operations.read_image(file)
            return img.shape[0]
        else:
            return self.__get_chromosome_len_using_curve_len(file)

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
        return min_position

    def dump_in_file_features(self):
        with open(self.features_out_file, "w") as f:
            f.write(str(len([_ for _ in self.features.keys()])) + "\n")
            if BANDAGE_PROFILE_KEY in self.FEATURE_USED_FOR_CLUSTERING:
                f.write(str(len(self.FEATURE_USED_FOR_CLUSTERING) - 1 + self.__nr_of_bandages_to_extract) + "\n")
            else:
                f.write(str(len(self.FEATURE_USED_FOR_CLUSTERING)) + "\n")
            for key in self.features:
                line = self.features[key][CHROMOSOME_NUMBER_KEY]
                line += ";"
                line += str(self.features[key][CHROMOSOME_COLOR_KEY][0]) + ":"
                line += str(self.features[key][CHROMOSOME_COLOR_KEY][1]) + ":"
                line += str(self.features[key][CHROMOSOME_COLOR_KEY][2]) + ";"
                line += key
                for feature in self.FEATURE_USED_FOR_CLUSTERING:
                    feature = feature[0]
                    if feature == BANDAGE_PROFILE_KEY:
                        line += ", " + ",".join([str(v) for v in self.features[key][feature]])
                    else:
                        line += ", " + str(self.features[key][feature])
                line += "\n"
                f.write(line)
        with open(self.features_out_json_file, "w") as fp:
            json.dump(self.features, fp, indent=4)
        with open(self.features_out_pickle_file, "wb") as fp:
            pickle.dump(self.features, fp)

        if len(self.FEATURE_USED_FOR_CLUSTERING) > 1 or (
                len(self.FEATURE_USED_FOR_CLUSTERING) == 1
                and self.FEATURE_USED_FOR_CLUSTERING[0][0] == BANDAGE_PROFILE_KEY):
            with open(self.features_weights_out_file, "w") as f:
                weights_string = ""
                for i in range(len(self.FEATURE_USED_FOR_CLUSTERING)):
                    feature = self.FEATURE_USED_FOR_CLUSTERING[i][0]
                    feature_weight = self.FEATURE_USED_FOR_CLUSTERING[i][1]
                    if feature == BANDAGE_PROFILE_KEY:
                        b_feature_weight = feature_weight / self.__nr_of_bandages_to_extract
                        for _ in range(self.__nr_of_bandages_to_extract):
                            weights_string += str(b_feature_weight) + ","
                    else:
                        weights_string += str(feature_weight) + ","
                f.write(weights_string[:-1])
        else:
            self.features_weights_out_file = ""

    def __get_chromosome_len_using_curve_len(self, img):
        medial_axis_obj = medial_axis.MedialAxis(img, self.__outputs_root_dir)
        curve_path = medial_axis_obj.get_medial_axis_img()
        if not medial_axis_len.Curve.is_valid_curve(curve_path):
            curve_path = medial_axis_obj.get_bin_inv_img_path()
        medial_axis_len_obj = medial_axis_len.Curve(curve_path, grade=CURVE_POLYNOMIAL_FUNCTION_GRADE)
        chromosome_length = medial_axis_len_obj.get_curve_length()
        return chromosome_length

    def __extract_area_feature(self):
        self.logger.debug("Start area feature extraction")
        for file in self.get_all_images():
            self.logger.debug("Extract area feature for %s" % os.path.basename(file))
            self.features[file][CHROMOSOME_AREA_KEY] = self.__get_chromosome_area(file)

    def get_chromosome_area(self, file):
        return self.__get_chromosome_area(file)

    def __get_chromosome_area(self, file):
        """
        :param file: straighten image file
        :return: chromosome area
        """
        colored_individual_ch_file_path = self.__get_individual_colored_path_from_individual_straighten_path(file)
        img = common_operations.read_image(colored_individual_ch_file_path)
        area = 0
        for i in img:
            for j in i:
                if any(j != 0):
                    area += 1
        return area

    @staticmethod
    def __get_individual_colored_path_from_individual_straighten_path(file):
        chromosome_nr = os.path.split(file)[1].split("-")[0]
        contrast_split_dir_path = os.path.split(os.path.split(file)[0])[0]
        return os.path.join(contrast_split_dir_path, chromosome_nr) + "_c.bmp"


def extract_features_from_imgs(ready_dir, root_dir, outs_dir, features=0x07, weights=None):
    obj = FeatureExtractor(ready_dir, root_dir, outs_dir, weights=weights)
    obj.extract_features(features)
    out_dir = os.path.join(ready_dir, "with_line")
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    return out_dir, obj.features_out_file, obj.features_weights_out_file
