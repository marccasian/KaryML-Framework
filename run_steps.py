import argparse
import datetime
import os

import sys

import time
from typing import List

from A_Segmentation.common_operations import get_all_images
from a_Common.my_logger import LOGGER, get_new_logger
from a_Common.constants import *

from A_Segmentation.main import ImageSegmenter

from B_Straightening.chromosome_orientation_updater import update as update_orientation
from C_FeatureExtraction.feature_extractor import extract_features_from_imgs
from D_PostProcessSOMResults.SOM_result_v2 import interpret_som_result
from D_PostProcessSOMResults.accuracy_calculator import compute_accuracy

logger = get_new_logger("run_steps_{}".format(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')))


def __init_arg_parser():
    parser = argparse.ArgumentParser(description='KarySOM')
    parser.add_argument('-f', type=str, nargs='+', default=[],
                        help='Path to chromosomes image')
    parser.add_argument('-d', type=str, nargs='+', default=[],
                        help='Path to chromosomes images dir')
    parser.add_argument('-ffile', type=str, nargs='?', default=[],
                        help='Path to features file')
    parser.add_argument('-wfile', type=str, nargs='?', default=[],
                        help='Path to weights file')
    parser.add_argument('-nfile', type=str, nargs='?', default="",
                        help='Path to neurons file')
    parser.add_argument('-mdist', type=str, nargs='?', default=[],
                        help='use manattan dist')
    parser.add_argument('-deserialize', type=bool, nargs='?', default=False,
                        help='Deserialize neurons from neurons file')
    parser.add_argument('-features', type=int, nargs='?', default=0x0f,
                        help='features to be extracted: '
                             '\n\t 0x01 for chromosome length'
                             '\n\t 0x02 for centromeric index'
                             '\n\t 0x04 for banding pattern'
                             '\n\t 0x08 for chromosome area'
                             '\n Combination are allowed ex: 0x03 for length and centromeric index'
                             '\n Default value is 0x0f (all 4 features)')
    parser.add_argument('-epochs', type=int, nargs='?', default=200000,
                        help='Number of epochs used for som training process')
    parser.add_argument('-rows', type=int, nargs='?', default=50,
                        help='Number of rows used for SOM map')
    parser.add_argument('-cols', type=int, nargs='?', default=50,
                        help='Number of cols used for SOM map')
    cmd_line_args = parser.parse_args()
    return cmd_line_args


def init_logger():
    current_logger = LOGGER.getChild("RunSteps")
    return current_logger


def update_gui_stat(gui, msg, error=False):
    msg = '\n'.join(msg[i:i + 78] for i in range(0, len(msg), 78))
    if gui is None:
        return
    if error:
        gui.errors_label.value = msg
        gui.errors_label.show()
    else:
        gui.info_label.value = msg
        gui.info_label.show()


def start_process_for_one_image(image_path, mdist, nfile="", features=0x07, epochs=200000, rows=50, cols=50,
                                fweights: List[float] = None,
                                gui=None):
    try:
        contrast_split_dir_path, individual_dir_path, root_dir_path = step_1_segmentation(image_path, gui)
        straight_dir_path = step_2_orientation_update(contrast_split_dir_path, image_path, individual_dir_path, gui)
        features_out_file_path, features_weight_out_file_path = step_3_feature_extraction(features, fweights,
                                                                                          image_path, root_dir_path,
                                                                                          straight_dir_path, gui)

        run_classifier_and_interpret_results(features_out_file_path, features_weight_out_file_path, root_dir_path,
                                             mdist, nfile, epochs, rows, cols, gui)
    except:
        __handle_process_exception(gui)


def run_classifier_and_interpret_results(features_out_file_path, features_weight_out_file_path, root_dir_path, mdist,
                                         nfile, epochs, rows, cols, gui=None, run_instance_start_timestamp_str=None,
                                         som_output_file_path=None, classifier_type=SOM_CLASSIFIER_TYPE):
    """

    :param classifier_type:
    :param som_output_file_path:
    :param run_instance_start_timestamp_str:
    :param cols:
    :param epochs:
    :param features_out_file_path:
    :param features_weight_out_file_path:
    :param gui:
    :param mdist:
    :param nfile:
    :param root_dir_path:
    :param rows:
    :return:
        model_output_file_path
        dist_matrix_file_path
        generated_karyotype_image_path
    """
    deserialize = check_neurons_file_existance(nfile)
    som_output_file = step_4_run_classifier(deserialize, features_out_file_path, features_weight_out_file_path,
                                            mdist, nfile, epochs, rows, cols, gui,
                                            som_output_file_path=som_output_file_path, classifier_type=classifier_type)
    dist_matrix_file_path, pairs_file_path, k_img = step_5_interpret_classifier_output(root_dir_path, som_output_file,
                                                                                       gui)
    preckar = None
    if os.path.isfile(pairs_file_path):
        preckar = step_6_preckar_computation(deserialize, dist_matrix_file_path, features_out_file_path, nfile,
                                             pairs_file_path, gui, timestamp_str=run_instance_start_timestamp_str)

    return som_output_file, dist_matrix_file_path, k_img, preckar


def __handle_process_exception(gui):
    msg = "Something went wrong"
    import traceback
    logger.error("Exception occurred %s" % traceback.format_exc())
    update_gui_stat(gui, msg, error=True)


def check_neurons_file_existance(nfile):
    if nfile is None:
        logger.debug("nfile %s don't exist, will disable deserialize" % nfile)
        deserialize = False
    elif not os.path.exists(nfile):
        logger.debug("nfile %s don't exist, will disable deserialize" % nfile)
        deserialize = False
    elif not os.path.isfile(nfile):
        logger.debug("nfile %s exist but not regular file, will disable deserialize" % nfile)
        deserialize = False
    else:
        deserialize = True
    return deserialize


def step_1_segmentation(image_path: str, gui=None):
    msg = "Start segmentation process for image: %s" % image_path
    logger.info(msg)
    update_gui_stat(gui, msg)
    img_segmenter_obj = ImageSegmenter()
    root_dir_path, individual_dir_path, contrast_split_dir_path = img_segmenter_obj.process_one_image(image_path)
    msg = "Done segmentation process for image: %s" % image_path + "\n" + "Start orientation updater process for image: %s" % image_path
    update_gui_stat(gui, msg)
    logger.info("Done segmentation process for image: %s" % image_path)
    return contrast_split_dir_path, individual_dir_path, root_dir_path


def step_2_orientation_update(contrast_split_dir_path: str, image_path: str, individual_dir_path: str, gui=None,
                              skip_if_exists=True):
    logger.info("Start orientation updater process for image: %s" % image_path)
    straight_dir_path = os.path.join(contrast_split_dir_path, "straight")
    if not os.path.isdir(straight_dir_path) or not skip_if_exists:
        update_orientation(individual_dir_path, straight_dir_path)
    msg = "Done orientation updater process for image: %s" % image_path + "\n" + "Start feature extraction process for image: %s" % image_path
    update_gui_stat(gui, msg)
    logger.info("Done orientation updater process for image: %s" % image_path)
    return straight_dir_path


def step_3_feature_extraction(features: int, fweights: List[float], image_path: str, root_dir_path: str,
                              straight_dir_path: str, gui=None):
    logger.info("Start feature extraction process for image: %s" % image_path)
    outs_dir_path = os.path.join(root_dir_path, "Outputs")
    line_dir_path, features_out_file_path, features_weight_out_file_path = extract_features_from_imgs(
        straight_dir_path,
        root_dir_path,
        outs_dir_path,
        features,
        fweights)
    msg = "Done feature extraction process for image: %s" % image_path + "\n Start SOM Classification"
    update_gui_stat(gui, msg)
    logger.info("Done feature extraction process for image: %s" % image_path)
    logger.debug("Features out file path: %s for image: %s" % (image_path, features_out_file_path))
    return features_out_file_path, features_weight_out_file_path


def step_4_run_classifier(deserialize: bool, features_out_file_path: str, features_weight_out_file_path: str,
                          use_manhatan_dist: bool, nfile: str, epochs: int, rows: int, cols: int, gui=None,
                          som_output_file_path=None, classifier_type=SOM_CLASSIFIER_TYPE):
    logger.debug("Start SOM Classification...")
    som_output_file = run_som(features_out_file_path, features_weight_out_file_path, use_manhatan_dist, nfile,
                              deserialize, epochs=epochs, rows=rows, cols=cols,
                              som_output_file_path=som_output_file_path, classifier_type=classifier_type)
    msg = "End SOM Classification \n Start SOM output interpretation"
    update_gui_stat(gui, msg)
    logger.debug("End SOM Classification")
    return som_output_file


def step_5_interpret_classifier_output(root_dir_path: str, som_output_file: str, gui=None):
    logger.debug("Start SOM output interpretation")
    pairs_file_path = os.path.join(root_dir_path, "pairs.txt")
    dist_matrix_file_path, k_img = interpret_som_result(som_output_file, pairs_file_path)
    msg = "End SOM output interpretation. \n Karyotype generated image path: %s" % k_img
    update_gui_stat(gui, msg)
    logger.debug("End SOM output interpretation")
    return dist_matrix_file_path, pairs_file_path, k_img


def step_6_preckar_computation(deserialize: bool, dist_matrix_file_path: str, features_out_file_path: str, nfile: str,
                               pairs_file_path: str, gui=None, timestamp_str=None):
    msg = "Compute accuracy ... "
    update_gui_stat(gui, msg)
    logger.debug("Compute accuracy ... ")
    preckar = compute_accuracy(pairs_file_path, dist_matrix_file_path, features_file=features_out_file_path,
                               neurons_file=nfile, deserialize=deserialize, timestamp_str=timestamp_str)
    logger.debug("Done accuracy computation")
    msg = "Done accuracy computation" + "\n Accuracy = {}".format(preckar)
    update_gui_stat(gui, msg)
    return preckar


def run_som(
        features_file=r'D:\GIT\Karyotyping-Project\PythonProject\Z_Images\autom\1\contrast_split\straight_features.txt',
        weights_file=r'D:\GIT\Karyotyping-Project\PythonProject\Z_Images\autom\1\contrast_split\straight_features_weights.txt',
        mdist=False,
        nfile="",
        deserialize=False,
        epochs=200000,
        rows=50,
        cols=50,
        som_output_file_path=None,
        classifier_type=SOM_CLASSIFIER_TYPE
):
    import subprocess
    # -aqe good_aqw_1.txt
    #  -features straight_features.txt -weights straight_features_weights.txt -e 200000 -rows 50 -cols 50
    logger = init_logger()
    som_path = "F-SOMv1.exe"
    if mdist:
        cmd_line = "%s -features %s -manhattan -e %d -r %d -c %d" \
                   % (som_path, features_file, epochs, rows, cols)
    elif weights_file != "":
        cmd_line = "%s -features %s -weights %s -e %d -r %d -c %d" \
                   % (som_path, features_file, weights_file, epochs, rows, cols)
    else:
        cmd_line = "%s -features %s -e %d -r %d -c %d" \
                   % (som_path, features_file, epochs, rows, cols)
    if som_output_file_path:
        cmd_line += " -out %s" % som_output_file_path
    if classifier_type == SOM_CLASSIFIER_TYPE:
        cmd_line += " -nofuzzy"
    cmd_line += " -neurons %s" % nfile
    logger.info("Deserialize = %s" % deserialize)
    if os.path.isfile(nfile) and deserialize:
        cmd_line += " -deserialize"
    logger.info("Start {} with following cmd line:".format(classifier_type))
    logger.info(cmd_line)

    if som_output_file_path is None:
        som_output_file_path = features_file + ".out"
    som_done_file_path = som_output_file_path + ".done"

    if os.path.exists(som_done_file_path):
        os.remove(som_done_file_path)
    proc = subprocess.Popen(cmd_line, stdout=sys.stdout, stderr=sys.stdout)
    while True:
        if os.path.isfile(som_done_file_path):
            logger.debug(
                "{} output done file detected, sleep 10 seconds and kill {} process...".format(classifier_type,
                                                                                               classifier_type))
            time.sleep(10)
            logger.debug("Kill {} process...".format(classifier_type))
            proc.kill()
            break
        else:
            logger.debug("{} still running...".format(classifier_type))
            time.sleep(5)

    return som_output_file_path


def start_som_and_acc_computation(ffile: str, wfile: str, use_manhattan_dist: bool, nfile: str, epochs: int, rows: int,
                                  cols: int):
    logger.debug("Start SOM Classification...")
    if not nfile:
        logger.debug("nfile don't exist, will serialize neurons in default file path")
        nfile = ""
        deserialize = False
    elif nfile is None:
        logger.debug("nfile is none, will serialize neurons in default file path")
        nfile = ""
        deserialize = False
    elif not os.path.isfile(nfile):
        logger.debug("nfile don't exist, will disable deserialize")
        deserialize = False
    else:
        deserialize = True
    if not wfile:
        wfile = ""
    if not os.path.exists(wfile):
        wfile = ""
    if not use_manhattan_dist:
        use_manhattan_dist = False

    som_output_file = run_som(ffile, wfile, use_manhattan_dist, nfile, deserialize, epochs=epochs, rows=rows, cols=cols)
    logger.debug("End SOM Classification")
    logger.debug("Start SOM output interpretation")
    pairs_file_path = os.path.join(os.path.dirname(os.path.dirname(ffile)), "pairs.txt")
    dist_matrix_file_path = interpret_som_result(som_output_file, pairs_file_path)
    logger.debug("End SOM output interpretation")
    if os.path.isfile(pairs_file_path):
        logger.debug("Compute accuracy ... ")
        compute_accuracy(pairs_file_path, dist_matrix_file_path, ffile, nfile, deserialize)
        logger.debug("Done accuracy computation")
    if os.path.isfile(os.path.join(os.path.dirname(ffile), "pairs.txt")):
        logger.debug("Compute accuracy ... ")
        compute_accuracy(os.path.join(os.path.dirname(ffile), "pairs.txt"), dist_matrix_file_path, ffile, nfile,
                         deserialize)
        logger.debug("Done accuracy computation")


if __name__ == "__main__":
    logger = init_logger()
    args = __init_arg_parser()
    try:
        logger.debug(args.f)
        logger.debug(args.d)
        logger.debug(args.features)
        logger.debug(args.ffile)
        logger.debug("args.wfile")
        logger.debug(args.wfile)
        logger.debug(args.mdist)
        logger.debug(args.deserialize)
        logger.debug(args.nfile)
        logger.debug(args.epochs)
        logger.debug(args.rows)
        logger.debug(args.cols)
        if len(args.ffile) > 0:
            if os.path.exists(args.ffile):
                start_som_and_acc_computation(args.ffile, args.wfile, args.mdist, args.nfile,
                                              args.epochs, args.rows, args.cols)
        logger.debug("Images:")
        for image in args.f:
            if not os.path.isfile(image):
                logger.error("""File "%s" doesn't exists""" % image)
            else:
                start_process_for_one_image(image_path=image, mdist=args.mdist, nfile=args.nfile,
                                            features=int(args.features), epochs=args.epochs, rows=args.rows,
                                            cols=args.cols)
        logger.debug("Dirs:")
        for dir_path in args.d:
            if not os.path.isdir(dir_path):
                logger.error("""Dir "%s" doesn't exists""" % dir_path)
            else:
                for image in get_all_images(dir_path):
                    start_process_for_one_image(image_path=image, mdist=args.mdist, nfile=args.nfile,
                                                features=int(args.features), epochs=args.epochs, rows=args.rows,
                                                cols=args.cols)
    except:
        import traceback

        logger.error("Traceback: %s" % traceback.format_exc())
