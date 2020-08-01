import datetime
import traceback

import math

from D_PostProcessSOMResults.expected_karyotype_loader import ExpectedKaryotype
from a_Common.my_logger import LOGGER
from D_PostProcessSOMResults.SOM_results_constants import *


class AccuracyCalculator:
    def __init__(self, pairs_file_path, dist_matrix_file_path):
        self.pairs_file_path = pairs_file_path
        self.dist_matrix_file_path = dist_matrix_file_path
        self.pairs = list()
        self.dist_matrix = list()
        self.exponential_scale_accuracy = list()
        self.normal_scale_accuracy = list()
        self.chromosomes_matching_lists = list()

    def __load_dist_matrix_from_file(self):
        """
            File format:
            n,m
            n lines and c columns
            :return:
        """
        try:
            with open(self.dist_matrix_file_path, "r") as f:
                content = f.read()
                lines = content.split("\n")
                first_line = lines[0]
                lines = lines[1:]
                n = int(first_line.split(",")[0].strip())
                m = int(first_line.split(",")[1].strip())
                self.dist_matrix = list()
                for i in range(0, n):
                    current_dist_matrix_line = list()
                    for elem in lines[i].split(CH_DIST_MATRIX_FILE_VALUES_SEPARATOR):
                        dist = float(elem.strip())
                        current_dist_matrix_line.append(dist)
                    if len(current_dist_matrix_line) == 0:
                        # skip blank lines
                        continue
                    if len(current_dist_matrix_line) != m:
                        raise ValueError("Invalid file format!")
                    self.dist_matrix.append(current_dist_matrix_line)
        except:
            print("Exception occurred while trying to load dist matrix from file. Traceback: %s"
                  % traceback.format_exc())

    def __load_data_from_files(self):
        self.__load_dist_matrix_from_file()
        self.pairs = ExpectedKaryotype(self.pairs_file_path).load()

    def get_accuracy(self):
        if self.pairs == list() or self.dist_matrix == list():
            self.__load_data_from_files()
        self.__compute_chromosomes_matching_lists()
        norm_scale_accuracy_values = self.__get_normal_scale_accuracy_values(len(self.dist_matrix))
        exp_scale_accuracy_values = self.__get_exponential_scale_accuracy_values(len(self.dist_matrix))
        self.exponential_scale_accuracy = [exp_scale_accuracy_values[len(exp_scale_accuracy_values) // 2]
                                           for _ in range(len(self.dist_matrix))]
        self.normal_scale_accuracy = [norm_scale_accuracy_values[len(norm_scale_accuracy_values) // 2]
                                      for _ in range(len(self.dist_matrix))]
        for pair in self.pairs:
            if len(pair) == 2:
                pair_1_0_index = self.chromosomes_matching_lists[pair[1]].index(pair[0])
                pair_0_1_index = self.chromosomes_matching_lists[pair[0]].index(pair[1])
                self.exponential_scale_accuracy[pair[0]] = exp_scale_accuracy_values[pair_0_1_index]
                self.normal_scale_accuracy[pair[0]] = norm_scale_accuracy_values[pair_0_1_index]
                self.exponential_scale_accuracy[pair[1]] = exp_scale_accuracy_values[pair_1_0_index]
                self.normal_scale_accuracy[pair[1]] = norm_scale_accuracy_values[pair_1_0_index]
            if len(pair) == 3:
                pair_0_1_index = self.chromosomes_matching_lists[pair[0]].index(pair[1])
                pair_0_2_index = self.chromosomes_matching_lists[pair[0]].index(pair[2])
                if pair_0_1_index > pair_0_2_index:
                    pair_0_1_index -= 1
                else:
                    pair_0_2_index -= 1
                pair_1_0_index = self.chromosomes_matching_lists[pair[1]].index(pair[0])
                pair_1_2_index = self.chromosomes_matching_lists[pair[1]].index(pair[2])
                if pair_1_0_index > pair_1_2_index:
                    pair_1_0_index -= 1
                else:
                    pair_1_2_index -= 1
                pair_2_0_index = self.chromosomes_matching_lists[pair[2]].index(pair[0])
                pair_2_1_index = self.chromosomes_matching_lists[pair[2]].index(pair[1])
                if pair_2_0_index > pair_2_1_index:
                    pair_2_0_index -= 1
                else:
                    pair_2_1_index -= 1

                self.exponential_scale_accuracy[pair[0]] = (exp_scale_accuracy_values[pair_0_1_index] +
                                                            exp_scale_accuracy_values[pair_0_2_index]) / 2
                self.normal_scale_accuracy[pair[0]] = (norm_scale_accuracy_values[pair_0_1_index] +
                                                       norm_scale_accuracy_values[pair_0_2_index]) / 2
                self.exponential_scale_accuracy[pair[1]] = (exp_scale_accuracy_values[pair_1_0_index] +
                                                            exp_scale_accuracy_values[pair_1_2_index]) / 2
                self.normal_scale_accuracy[pair[1]] = (norm_scale_accuracy_values[pair_1_0_index] +
                                                       norm_scale_accuracy_values[pair_1_2_index]) / 2
                self.exponential_scale_accuracy[pair[2]] = (exp_scale_accuracy_values[pair_2_0_index] +
                                                            exp_scale_accuracy_values[pair_2_1_index]) / 2
                self.normal_scale_accuracy[pair[2]] = (norm_scale_accuracy_values[pair_2_0_index] +
                                                       norm_scale_accuracy_values[pair_2_1_index]) / 2

    def __compute_chromosomes_matching_lists(self):
        for i in range(len(self.dist_matrix)):
            self.chromosomes_matching_lists.append(self.__get_chromosome_matching_list(i))

    def __get_chromosome_matching_list(self, ch_index):
        a = sorted(range(len(self.dist_matrix[ch_index])), key=lambda x: self.dist_matrix[ch_index][x])
        a.remove(ch_index)
        return a

    @staticmethod
    def __get_exponential_scale_accuracy_values(instances):
        step = math.log(100) / instances
        val = [100 - math.exp(i * step) for i in range(instances)]
        val[0] = 100
        return val

    @staticmethod
    def __get_normal_scale_accuracy_values(instances):
        step = 100 / instances
        return [100 - i * step for i in range(instances)]


def init_logger():
    current_logger = LOGGER.getChild("Accuracy Calculator")
    return current_logger


def compute_accuracy(pairs_file,
                     dist_matrix_file,
                     features_file="",
                     neurons_file="",
                     deserialize=False,
                     timestamp_str=None):
    logger = init_logger()
    obj = AccuracyCalculator(pairs_file, dist_matrix_file)
    import os
    acc_dir = os.path.join(os.path.dirname(pairs_file), "Notes")
    if timestamp_str:
        acc_file = os.path.join(acc_dir, '%s.acc' % (timestamp_str.replace(" ", "_").replace(":", "-")))
    else:
        acc_file = os.path.join(acc_dir, '%s.acc' % (datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')))
    if not os.path.exists(acc_dir):
        os.makedirs(acc_dir)

    obj.get_accuracy()
    with open(acc_file, 'w') as f:
        __triple_log_msg('\n'.join([str(i) for i in obj.pairs]), f, logger)
        __triple_log_msg("=============================\n", f, logger)
        __triple_log_msg('\n'.join([str(i) for i in enumerate(obj.chromosomes_matching_lists)]), f, logger)
        __triple_log_msg("=============================\n", f, logger)
        __triple_log_msg('\n'.join([str(i) for i in enumerate(obj.exponential_scale_accuracy)]), f, logger)

        preckar = sum(obj.exponential_scale_accuracy) / float(len(obj.exponential_scale_accuracy))
        avg_str = str(preckar)
        __triple_log_msg("neurons file: %s\n" % neurons_file, f, logger)
        __triple_log_msg("features file: %s\n" % features_file, f, logger)
        __triple_log_msg("deserialize: %s\n" % str(deserialize), f, logger)
        __triple_log_msg("AVG = %s\n" % avg_str, f, logger)
        if os.path.exists(neurons_file) and os.path.exists(features_file) and not deserialize and timestamp_str is None:
            try:
                os.rename(neurons_file, features_file + "_%s.neurons" % avg_str.replace(".", "_"))
                __triple_log_msg("Successfully_ renamed neurons file", f, logger)
            except:
                __triple_log_msg("Failed to rename neurons file using os.rename to features_file_path_acc.neurons, "
                                 "will try to add current date_time to neurons file name", f, logger)
                os.rename(neurons_file, features_file + "_%s_%s.neurons"
                          % (avg_str.replace(".", "_"), datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')))
        return preckar


def __triple_log_msg(msg, f, logger):
    print(msg)
    logger.info(msg)
    f.write(msg)
