import json
import os
import pprint
import sys
import traceback

from typing import List, Tuple

from scipy import randn
from scipy.stats import pearsonr, spearmanr, kendalltau

from C_FeatureExtraction.feature_extractions_constants import *
from D_PostProcessSOMResults.expected_karyotype_loader import ExpectedKaryotype
from a_Common.my_logger import get_new_logger


class DataSetLoader:
    TARGET_CLASS_TAG = "target"
    CHROMOSOME_IMAGE_PATH_KEY = "ch_paths"

    FEATURES_TAG_LIST = [CHROMOSOME_LEN_KEY, SHORT_CHROMATID_RATIO_KEY, CHROMOSOME_AREA_KEY]  # , BANDAGE_PROFILE_KEY]

    JSON_FEATURES_FILE_INDEX = 0
    EXPECTED_KARYOTYPE_FILE_INDEX = 1

    def __init__(self, ds_files: List[Tuple[str, str]]):
        """

        :param ds_files: list containing tuples:
            (<path to features json>, <path to expected karyotype>)
        """
        self.__ds_files: List[Tuple[str, str]] = ds_files
        self.ds = dict()
        self.logger = get_new_logger(self.__class__.__name__)
        self.__load_from_files()

    def __load_from_files(self):
        """
        Each entry has following format:
            <chromosome image path>: {
                "nr": str,  # chromosome key
                "color": List[int], #RGB
                "chromosome_len": float,
                "short_chromatid_ratio": float,
                "centromer_position": float,
                "best_image_path": str,
                "bandage_profile": List[float],
                "chromosome_area": int,
                "target": int
            },
        :return:
        """
        for ds_files_entry in self.__ds_files:
            self.logger.debug("Start loading {}".format(ds_files_entry))
            current_expected_karyotype: ExpectedKaryotype = ExpectedKaryotype(
                ds_files_entry[self.EXPECTED_KARYOTYPE_FILE_INDEX])
            current_expected_karyotype.load()

            with open(ds_files_entry[self.JSON_FEATURES_FILE_INDEX], "r") as f:
                current_features = json.load(f)
                for item in current_features:
                    self.ds[item] = current_features[item]
                    self.ds[item][self.TARGET_CLASS_TAG] = current_expected_karyotype.get_pair_id(
                        int(self.ds[item][CHROMOSOME_NUMBER_KEY]))

    def get_dataset_for_correlation(self) -> dict:
        new_ds_form = {
            self.CHROMOSOME_IMAGE_PATH_KEY: list(),
            CHROMOSOME_NUMBER_KEY: list(),
            CHROMOSOME_COLOR_KEY: list(),
            CHROMOSOME_LEN_KEY: list(),
            SHORT_CHROMATID_RATIO_KEY: list(),
            CENTROMERE_POSITION_KEY: list(),
            BEST_IMAGE_PATH_KEY: list(),
            BANDAGE_PROFILE_KEY: list(),
            CHROMOSOME_AREA_KEY: list(),
            self.TARGET_CLASS_TAG: list(),
        }
        for item in self.ds:
            new_ds_form[self.CHROMOSOME_IMAGE_PATH_KEY].append(item)
            new_ds_form[CHROMOSOME_NUMBER_KEY].append(self.ds[item][CHROMOSOME_NUMBER_KEY])
            new_ds_form[CHROMOSOME_COLOR_KEY].append(self.ds[item][CHROMOSOME_COLOR_KEY])
            new_ds_form[CHROMOSOME_LEN_KEY].append(self.ds[item][CHROMOSOME_LEN_KEY])
            new_ds_form[SHORT_CHROMATID_RATIO_KEY].append(self.ds[item][SHORT_CHROMATID_RATIO_KEY])
            new_ds_form[CENTROMERE_POSITION_KEY].append(self.ds[item][CENTROMERE_POSITION_KEY])
            new_ds_form[BEST_IMAGE_PATH_KEY].append(self.ds[item][BEST_IMAGE_PATH_KEY])
            new_ds_form[BANDAGE_PROFILE_KEY].append(self.ds[item][BANDAGE_PROFILE_KEY])
            new_ds_form[CHROMOSOME_AREA_KEY].append(self.ds[item][CHROMOSOME_AREA_KEY])
            new_ds_form[self.TARGET_CLASS_TAG].append(self.ds[item][self.TARGET_CLASS_TAG])
        return new_ds_form


class Correlation:

    def __init__(self, ds_files: List[Tuple[str, str]], function_to_call=None):
        """

        :param ds_files: list containing tuples:
            (<path to features json>, <path to expected karyotype>)
        """
        self.ds_loader = DataSetLoader(ds_files)
        self.logger = get_new_logger(self.__class__.__name__)
        self.ds: dict = self.ds_loader.get_dataset_for_correlation()
        self.function_to_call = function_to_call

    def compute(self):
        if callable(self.function_to_call):
            correlation_matrix = list()
            correlation_matrix.append([0] + self.ds_loader.FEATURES_TAG_LIST + [self.ds_loader.TARGET_CLASS_TAG])
            for feature in self.ds_loader.FEATURES_TAG_LIST + [self.ds_loader.TARGET_CLASS_TAG]:
                new_line = [feature]
                for i in range(len(self.ds_loader.FEATURES_TAG_LIST) + 1):
                    new_line.append(None)
                correlation_matrix.append(new_line)
            for i in range(1, len(self.ds_loader.FEATURES_TAG_LIST) + 2):
                for j in range(i, len(self.ds_loader.FEATURES_TAG_LIST) + 2):
                    if i == j:
                        current_corr = 1
                    else:
                        self.logger.debug("Computing pearson correlation between {} and {}".format(
                            correlation_matrix[j][0], correlation_matrix[0][i]))
                        current_corr = None
                        if correlation_matrix[j][0] == BANDAGE_PROFILE_KEY:
                            sum_corr = 0
                            for index in range(len(self.ds[correlation_matrix[j][0]][0])):
                                sum_corr += self.function_to_call([a[index] for a in self.ds[correlation_matrix[j][0]]],
                                                                  self.ds[correlation_matrix[0][i]])[0]
                            current_corr = sum_corr / len(self.ds[correlation_matrix[j][0]])
                        elif correlation_matrix[0][i] == BANDAGE_PROFILE_KEY:
                            sum_corr = 0
                            for index in range(len(self.ds[correlation_matrix[0][i]][0])):
                                sum_corr += self.function_to_call([a[index] for a in self.ds[correlation_matrix[0][i]]],
                                                                  self.ds[correlation_matrix[j][0]])[0]
                            current_corr = sum_corr / len(self.ds[correlation_matrix[0][i]])
                        else:
                            current_corr, _ = self.function_to_call(self.ds[correlation_matrix[j][0]],
                                                                    self.ds[correlation_matrix[0][i]])
                    correlation_matrix[i][j] = current_corr
                    correlation_matrix[j][i] = current_corr
            return correlation_matrix
        else:
            raise NotImplementedError("Please implement this correlation computation method according to "
                                      "class specification")


class PearsonCorrelation(Correlation):

    def __init__(self, ds_files: List[Tuple[str, str]]):
        super().__init__(ds_files, pearsonr)


class SpearmanCorrelation(Correlation):
    def __init__(self, ds_files: List[Tuple[str, str]]):
        super().__init__(ds_files, spearmanr)


# kendalltau


class KendalltauCorrelation(Correlation):
    def __init__(self, ds_files: List[Tuple[str, str]]):
        super().__init__(ds_files, kendalltau)



def correlation_computation(ds_files):
    cor_obj = KendalltauCorrelation(ds_files)
    correlations = cor_obj.compute()
    with open(ds_files[0][0] + "_kendalltau_correlation_no_bandage.txt", "w") as g:
        for i in correlations:
            for j in i:
                to_print = str(j)
                if len(to_print) < 25:
                    to_print = str(to_print) + " " * (25 - len(to_print))
                g.write('%.35s' % to_print)
                print('%.35s' % to_print, end="")
            g.write("\n")
            print()
    #
    #
    # # prepare data
    # data1 = 20 * randn(1000) + 100
    # data2 = data1 + (10 * randn(1000) + 50)
    # # calculate Pearson's correlation
    # corr, _ = pearsonr(data1, data2)
    # print('Pearsons correlation: %.3f' % corr)
    # ExpectedKaryotype("assd")


if __name__ == '__main__':
    ds_files = [
        (r'__disertation_experiments\dataset\1\straight_features.json',
         r'__disertation_experiments\dataset\1\1_expected_karyotype.txt'),
        (r'__disertation_experiments\dataset\3\straight_features.json',
         r'__disertation_experiments\dataset\3\3_expected_karyotype.txt'),
        (r'__disertation_experiments\dataset\5\straight_features.json',
         r'__disertation_experiments\dataset\5\5_expected_karyotype.txt'),
        (r'__disertation_experiments\dataset\6\straight_features.json',
         r'__disertation_experiments\dataset\6\6_expected_karyotype.txt'),
        (r'__disertation_experiments\dataset\7\straight_features.json',
         r'__disertation_experiments\dataset\7\7_expected_karyotype.txt'),
        (r'__disertation_experiments\dataset\8\straight_features.json',
         r'__disertation_experiments\dataset\8\8_expected_karyotype.txt'),
    ]

    length_plotter(ds_files)
