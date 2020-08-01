import argparse
import json
import os
import pickle
import sys
import traceback
from typing import List

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from C_FeatureExtraction.feature_extractor import FeatureExtractor
from F_Experiments_Helper.run_instance import RunInstance
from a_Common.my_logger import get_new_logger, LOGGER
from run_steps import run_classifier_and_interpret_results

DS_ENTRIES = [1, 3, 5, 6, 7, 8]

from F_Experiments_Helper.db import INPUT_IMAGE_PATH_COLUMN_NAME, FEATURES_FILE_PATH_COLUMN_NAME, delete_run, \
    update_run_entry, insert_new_run, get_best_run_so_far_with_similar_input, how_many_similar_input_runs_so_far
from a_Common.constants import *

from C_FeatureExtraction.feature_extractions_constants import *


def init_logger():
    current_logger = LOGGER.getChild("ClassifierRunner")
    return current_logger


logger = init_logger()

DS_ROOT_DIR_PATH = r'__disertation_experiments\dataset'
DS_ENTRY_DIR_WORKING_DIR_ROOT_PATH = "root_dir_path"
INPUT_FILES_ROOT_DIR_NAME = "inputs"
OUTPUT_FILES_ROOT_DIR_NAME = "outputs"

CLASSIFIER_EXE_PATH_DICT = {
    SOM_CLASSIFIER_TYPE: "F-SOMv1.exe",
    FSOM_CLASSIFIER_TYPE: "F-SOMv1.exe"
}


class ClassifierReqFilesGenerator:
    # where the parameter represents the features hexa encoding as int
    FEATURES_FILE_NAME_MASK = "features_{}.txt"

    # where the first parameter represents the features hexa encoding as int and the second one its a _ separated
    # values representing the int value of each feature weight
    FEATURES_WEIGHTS_FILE_NAME_MASK = "weights_{}{}.txt"

    def __init__(self, ds_descriptor: dict):
        self.ds_descriptor = ds_descriptor
        self.logger = get_new_logger(self.__class__.__name__)

    def generate_input_feature_files(self, ds_entry: str, features: int = 0x0f, weights: List[float] = None):
        """
        :param ds_entry: str din 1,3,5,6,7,8
        :param features: 0x0...
        :param weights: list containing used feature weights in case of weighted euclidean
        :return:
        """
        all_features_json_file_path = self.ds_descriptor[ds_entry][FEATURES_FILE_PATH_COLUMN_NAME]
        all_features = dict()
        with open(all_features_json_file_path, "r") as f:
            all_features = json.load(f)
        input_image_path = self.ds_descriptor[ds_entry][INPUT_IMAGE_PATH_COLUMN_NAME]
        working_dir_root_path = self.ds_descriptor[ds_entry][DS_ENTRY_DIR_WORKING_DIR_ROOT_PATH]
        input_files_root_dir_path = os.path.join(os.path.dirname(input_image_path), INPUT_FILES_ROOT_DIR_NAME)
        output_files_root_dir_path = os.path.join(os.path.dirname(input_image_path), OUTPUT_FILES_ROOT_DIR_NAME)
        os.makedirs(input_files_root_dir_path, exist_ok=True)
        os.makedirs(output_files_root_dir_path, exist_ok=True)

        new_feature_file_path = os.path.join(input_files_root_dir_path, self.FEATURES_FILE_NAME_MASK.format(features))
        features_weights_file_name_ending = ""
        if weights:
            features_weights_file_name_ending = "__" + "_".join([str(int(i)) for i in weights])
        new_weights_file_path = os.path.join(input_files_root_dir_path, self.FEATURES_WEIGHTS_FILE_NAME_MASK.format(
            features, features_weights_file_name_ending))
        new_weights_file_path, feature_used_list = self.dump_in_files(features, all_features, new_feature_file_path,
                                                                      weights,
                                                                      new_weights_file_path)
        return self.ds_descriptor[ds_entry][DS_ENTRY_DIR_WORKING_DIR_ROOT_PATH], new_feature_file_path, \
               new_weights_file_path, feature_used_list, output_files_root_dir_path

    def dump_in_files(self, features_mask, features_dict, new_feature_file_path, weights,
                      new_weights_file_path):
        feature_used, number_of_bandages = self.build_feature_used_list(features_dict, features_mask, weights)

        self.dump_features_file(feature_used, features_dict, new_feature_file_path, number_of_bandages)

        if len(feature_used) > 1 or (
                len(feature_used) == 1
                and feature_used[0][0] == BANDAGE_PROFILE_KEY):
            self.dump_weights_file(feature_used, new_weights_file_path, number_of_bandages)
        else:
            new_weights_file_path = ""
        return new_weights_file_path, feature_used

    @staticmethod
    def dump_features_file(feature_used, features_dict, new_feature_file_path, number_of_bandages):
        if os.path.exists(new_feature_file_path):
            return
        with open(new_feature_file_path, "w") as f:
            f.write(str(len([_ for _ in features_dict.keys()])) + "\n")
            if BANDAGE_PROFILE_KEY in feature_used:
                f.write(str(len(feature_used) - 1 + number_of_bandages) + "\n")
            else:
                f.write(str(len(feature_used)) + "\n")
            for key in features_dict:
                line = features_dict[key][CHROMOSOME_NUMBER_KEY]
                line += ";"
                line += str(features_dict[key][CHROMOSOME_COLOR_KEY][0]) + ":"
                line += str(features_dict[key][CHROMOSOME_COLOR_KEY][1]) + ":"
                line += str(features_dict[key][CHROMOSOME_COLOR_KEY][2]) + ";"
                line += key
                for feature in feature_used:
                    feature = feature[0]
                    if feature == BANDAGE_PROFILE_KEY:
                        line += ", " + ",".join([str(v) for v in features_dict[key][feature]])
                    else:
                        line += ", " + str(features_dict[key][feature])
                line += "\n"
                f.write(line)

    @staticmethod
    def dump_weights_file(feature_used, new_weights_file_path, number_of_bandages):
        if os.path.exists(new_weights_file_path):
            return
        with open(new_weights_file_path, "w") as f:
            weights_string = ""
            for i in range(len(feature_used)):
                feature = feature_used[i][0]
                feature_weight = feature_used[i][1]
                if feature == BANDAGE_PROFILE_KEY:
                    b_feature_weight = feature_weight / number_of_bandages
                    for _ in range(number_of_bandages):
                        weights_string += str(b_feature_weight) + ","
                else:
                    weights_string += str(feature_weight) + ","
            f.write(weights_string[:-1])

    @staticmethod
    def build_feature_used_list(features_dict, features_mask, weights):
        number_of_bandages = None
        feature_used = list()
        features_no = sum([1 for _ in [1 for i in [FeatureExtractor.CHROMOSOME_LEN_FEATURE_FLAG,
                                                   FeatureExtractor.CHROMOSOME_SHORT_CHROMATID_RATIO_FEATURE_FLAG,
                                                   FeatureExtractor.CHROMOSOME_BANDAGE_PROFILE_FEATURE_FLAG,
                                                   FeatureExtractor.CHROMOSOME_CHROMOSOME_AREA_FEATURE_FLAG]
                                       if features_mask & i != 0]])
        weights_index = 0
        if features_mask & FeatureExtractor.CHROMOSOME_LEN_FEATURE_FLAG != 0:
            if weights is not None:
                feature_used.append([CHROMOSOME_LEN_KEY, weights[weights_index]])
                weights_index += 1
            else:
                feature_used.append([CHROMOSOME_LEN_KEY, 1 / features_no])
        if features_mask & FeatureExtractor.CHROMOSOME_SHORT_CHROMATID_RATIO_FEATURE_FLAG != 0:
            if weights is not None:
                feature_used.append([SHORT_CHROMATID_RATIO_KEY, weights[weights_index]])
                weights_index += 1
            else:
                feature_used.append([SHORT_CHROMATID_RATIO_KEY, 1 / features_no])
        if features_mask & FeatureExtractor.CHROMOSOME_BANDAGE_PROFILE_FEATURE_FLAG != 0:
            number_of_bandages = len(features_dict[list(features_dict.keys())[0]][BANDAGE_PROFILE_KEY])
            if weights is not None:
                feature_used.append([BANDAGE_PROFILE_KEY, weights[weights_index]])
                weights_index += 1
            else:
                feature_used.append([BANDAGE_PROFILE_KEY, 1 / features_no])
        if features_mask & FeatureExtractor.CHROMOSOME_CHROMOSOME_AREA_FEATURE_FLAG != 0:
            if weights is not None:
                feature_used.append([CHROMOSOME_AREA_KEY, weights[weights_index]])
                weights_index += 1
            else:
                feature_used.append([CHROMOSOME_AREA_KEY, 1 / features_no])
        return feature_used, number_of_bandages


def determine_distance_type(mdist, weights):
    if mdist:
        return MANHATTAN_DISTANCE
    if weights:
        return WEIGHTED_EUCLIDEAN_DISTANCE
    return EUCLIDEAN_DISTANCE


def generate_neurons_file_path(outs_last_dir_path):
    return os.path.join(outs_last_dir_path, "java_neurons.obj")


def generate_model_output_file_path(features_file_path, outputs_dir_path, run_instance_obj):
    last_dir_path = os.path.join(os.path.join(outputs_dir_path, os.path.basename(features_file_path)[:-4]),
                                 run_instance_obj.start_time.replace(" ", "_").replace(":", "-"))
    os.makedirs(last_dir_path, exist_ok=True)
    return os.path.join(last_dir_path, "classifier.out"), last_dir_path


def main(ds_entry: str, features: int, weights: List[float], nfile: str, mdist: bool, epochs: int, rows: int,
         cols: int, classifier_type: str):
    ds_descriptor = None
    with open(os.path.join(DS_ROOT_DIR_PATH, "description.json"), "r") as f:
        ds_descriptor = json.load(f)

    file_generator_obj = ClassifierReqFilesGenerator(ds_descriptor=ds_descriptor)

    root_dir_path, features_file_path, features_weight_file_path, feature_used_list, outputs_dir_path = \
        file_generator_obj.generate_input_feature_files(ds_entry=ds_entry, features=features, weights=weights)
    distance = determine_distance_type(mdist, weights)

    run_instance_obj = RunInstance(classifier_type=classifier_type,
                                   input_image_path=ds_descriptor[ds_entry][INPUT_IMAGE_PATH_COLUMN_NAME],
                                   features_file_path=features_file_path, used_features=features, distance=distance,
                                   epochs=epochs, rows=rows, cols=cols,
                                   len_feature_weight=get_feature_weight(feature_used_list, CHROMOSOME_LEN_KEY),
                                   short_chromatid_ratio_feature_weight=get_feature_weight(feature_used_list,
                                                                                           SHORT_CHROMATID_RATIO_KEY),
                                   banding_pattern_feature_weights=get_feature_weight(feature_used_list,
                                                                                      BANDAGE_PROFILE_KEY),
                                   area_feature_weight=get_feature_weight(feature_used_list, CHROMOSOME_AREA_KEY))
    model_output_file_path, outs_last_dir_path = generate_model_output_file_path(features_file_path, outputs_dir_path,
                                                                                 run_instance_obj)
    best_run_so_far = get_best_run_so_far_with_similar_input(run_instance_obj, logger)

    how_many_runs_executed = how_many_similar_input_runs_so_far(run_instance_obj, logger, 5179)
    if how_many_runs_executed >= 30:
        logger.info(
            "Already executed {} similar runs for current cfg: {}".format(how_many_runs_executed, best_run_so_far))
        from a_Common.alarm import ring_sms
        ring_sms()
        return
    else:
        logger.info("{} similar runs executed so far".format(how_many_runs_executed))

    if nfile == "":
        initial_neurons_file = ""
        if best_run_so_far:
            if (os.path.exists(best_run_so_far.initial_neurons_file) and os.path.isfile(
                    best_run_so_far.initial_neurons_file)):
                initial_neurons_file = best_run_so_far.initial_neurons_file
            else:
                renamed_path = best_run_so_far.initial_neurons_file.replace("outputs", "{}_needed_outs".format(
                    classifier_type.lower()))
                if os.path.exists(renamed_path) and os.path.isfile(renamed_path):
                    initial_neurons_file = renamed_path
        if os.path.exists(initial_neurons_file):
            nfile = initial_neurons_file
            logger.info("Using initial neurons from a run having preckar {}".format(best_run_so_far.preckar))
        else:
            nfile = generate_neurons_file_path(outs_last_dir_path)

    run_instance_obj.model_output_file_path = model_output_file_path
    run_instance_obj.initial_neurons_file = nfile
    logger.debug("Pre run instance :{}".format(run_instance_obj))

    run_id = insert_new_run(run_instance_obj)
    run_instance_obj.id = run_id
    try:
        model_output_file_path, dist_matrix_file_path, generated_karyotype_image_path, preckar = \
            run_classifier_and_interpret_results(features_out_file_path=run_instance_obj.features_file_path,
                                                 features_weight_out_file_path=features_weight_file_path,
                                                 root_dir_path=root_dir_path, mdist=mdist,
                                                 nfile=run_instance_obj.initial_neurons_file,
                                                 epochs=run_instance_obj.epochs, rows=run_instance_obj.rows,
                                                 cols=run_instance_obj.cols,
                                                 run_instance_start_timestamp_str=run_instance_obj.start_time,
                                                 som_output_file_path=model_output_file_path,
                                                 classifier_type=run_instance_obj.classifier_type)
        run_instance_obj.model_output_file_path = model_output_file_path
        run_instance_obj.dist_matrix_file_path = dist_matrix_file_path
        run_instance_obj.generated_karyotype_image_path = generated_karyotype_image_path
        run_instance_obj.preckar = preckar
        if preckar >= 87.5:
            from a_Common.alarm import ring_pew
            ring_pew()
        run_instance_obj.set_end_time()
        update_run_entry(run_instance_obj)
        logger.debug("Post run instance :{}".format(run_instance_obj))
    except BaseException as exc:
        logger.exception(exc)
        logger.exception(traceback)
        delete_run(run_instance_obj)
    return root_dir_path, features_file_path, features_weight_file_path


def get_feature_weight(feature_used_list, feature):
    weight = None
    for f in feature_used_list:
        if f[0] == feature:
            weight = f[1]
            break
    return weight


def __init_arg_parser():
    parser = argparse.ArgumentParser(description='Classifier runner')
    parser.add_argument('-ds_entry', type=str, nargs='?', default="1",
                        help='Data set entry. ATM one of ["1", "3", "5", "6", "7", "8"]')
    parser.add_argument('-features', type=int, nargs='?', default=0x0f,
                        help='Features to be extracted: '
                             '\n\t 0x01 for chromosome length'
                             '\n\t 0x02 for centromeric index'
                             '\n\t 0x04 for banding pattern'
                             '\n\t 0x08 for chromosome area'
                             '\n Combination are allowed ex: 0x03 for length and centromeric index'
                             '\n Default value is 0x0f (all 4 features)')
    parser.add_argument('-classifier', type=str, nargs='?', default="SOM",
                        help='Classifier type. ATM one of: SOM or FSOM')
    parser.add_argument('-weights', type=float, nargs='*', default=None,
                        help='Weights array for given features')
    parser.add_argument('-nfile', type=str, nargs='?', default="",
                        help='Path to neurons file')
    parser.add_argument('-mdist', type=bool, nargs='?', default=False,
                        help='Use manhattan distance')
    parser.add_argument('-epochs', type=int, nargs='?', default=200000,
                        help='Number of epochs used for som training process')
    parser.add_argument('-rows', type=int, nargs='?', default=50,
                        help='Number of rows used for SOM map')
    parser.add_argument('-cols', type=int, nargs='?', default=50,
                        help='Number of cols used for SOM map')
    cmd_line_args = parser.parse_args()
    return cmd_line_args


if __name__ == '__main__':
    args = __init_arg_parser()
    try:
        logger.debug("args.ds_entry={}".format(args.ds_entry))
        logger.debug("args.features={}".format(args.features))
        logger.debug("args.weights={}".format(args.weights))
        logger.debug("args.nfile={}".format(args.nfile))
        logger.debug("args.mdist={}".format(args.mdist))
        logger.debug("args.epochs={}".format(args.epochs))
        logger.debug("args.rows={}".format(args.rows))
        logger.debug("args.cols={}".format(args.cols))
        logger.debug("args.classifier={}".format(args.classifier))
        main(args.ds_entry, args.features, args.weights, args.nfile, args.mdist, args.epochs, args.rows, args.cols,
             args.classifier)
    except BaseException as exc:
        logger.exception(exc)
        logger.exception(traceback.format_exc())
