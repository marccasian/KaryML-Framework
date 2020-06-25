from datetime import datetime
from typing import List

from a_Common.constants import *

DB_DATETIME_STR_FORMAT = '%Y-%m-%d %H:%M:%S'


class RunInstance:
    """
    # metadata columns
    ID_COLUMN_NAME = "id"
    START_TIME_COLUMN_NAME = "start_time"
    END_TIME_COLUMN_NAME = "end_time"

    # input columns
    CLASSIFIER_TYPE_COLUMN_NAME = "classifier_type"  # SOM or FSOM atm
    INPUT_IMAGE_PATH_COLUMN_NAME = "input_image_path"
    FEATURES_FILE_PATH_COLUMN_NAME = "features_file_path"
    USED_FEATURES_COLUMN_NAME = "used_features"  # 0x01 len, 0x02 - centromere, 0x04 banding, 0x08 area
    DISTANCE_COLUMN_NAME = "distance"  # euclidean, weighted - euclidean, manhattan
    LEN_FEATURE_WEIGHT_COLUMN_NAME = "len_feature_weight"  # weight for len feature
    SHORT_CHROMATID_RATIO_FEATURE_WEIGHT_COLUMN_NAME = "short_chromatid_ratio_feature_weight"  # weight centromere feature
    BANDING_PATTERN_FEATURE_WEIGHTS_COLUMN_NAME = "banding_pattern_feature_weights"  # weights banding pattern feature
    AREA_FEATURE_WEIGHT_COLUMN_NAME = "area_feature_weight"  # weight for area feature
    INITIAL_NEURONS_FILE_COLUMN_NAME = "used_neurons_file"  # initial model configuration (either existing one, meaning that the model starts from a given config, either the initial neurons of the model will be saved
    into this file)
    EPOCHS_COLUMN_NAME = "epochs"  # number of training epochs for the model
    ROWS_COLUMN_NAME = "rows"
    COLS_COLUMN_NAME = "cols"

    # outs
    MODEL_OUTPUT_FILE_PATH_COLUMN_NAME = "model_output_file_path"  # som output file
    DIST_MATRIX_FILE_PATH_COLUMN_NAME = "dist_matrix_file_path"
    GENERATED_KARYOTYPE_IMAGE_PATH = "generated_karyotype_image_path"
    PRECKAR_COLUMN_NAME = "preckar"
    """

    def __init__(self, classifier_type: str, input_image_path: str, features_file_path: str,
                 used_features: int, distance: str, epochs: int, rows: int, cols: int,
                 model_output_file_path: str = "", dist_matrix_file_path: str = "",
                 generated_karyotype_image_path: str = "",
                 preckar: float = None, id: int = None,
                 start_time: str = datetime.strftime(datetime.now(), DB_DATETIME_STR_FORMAT), end_time: str = None,
                 len_feature_weight: float = None, short_chromatid_ratio_feature_weight: float = None,
                 banding_pattern_feature_weights: float = None,
                 area_feature_weight: float = None, initial_neurons_file: str = None):
        self.id = id
        self.start_time = start_time
        self.end_time = end_time
        self.classifier_type = classifier_type
        self.input_image_path = input_image_path
        self.features_file_path = features_file_path
        self.used_features = used_features
        self.distance = distance
        self.len_feature_weight = len_feature_weight
        self.short_chromatid_ratio_feature_weight = short_chromatid_ratio_feature_weight
        self.banding_pattern_feature_weights = banding_pattern_feature_weights
        self.area_feature_weight = area_feature_weight
        self.initial_neurons_file = initial_neurons_file
        self.epochs = epochs
        self.rows = rows
        self.cols = cols
        self.model_output_file_path = model_output_file_path
        self.dist_matrix_file_path = dist_matrix_file_path
        self.generated_karyotype_image_path = generated_karyotype_image_path
        self.preckar = preckar

    def get_update_tuple(self):
        return (self.start_time, self.end_time, self.classifier_type, self.input_image_path,
                self.features_file_path, self.used_features, self.distance, self.len_feature_weight,
                self.short_chromatid_ratio_feature_weight, self.banding_pattern_feature_weights,
                self.area_feature_weight, self.initial_neurons_file, self.epochs, self.rows, self.cols,
                self.model_output_file_path, self.dist_matrix_file_path, self.generated_karyotype_image_path,
                self.preckar, self.id)

    def get_insert_tuple(self):
        return (self.start_time, self.end_time, self.classifier_type, self.input_image_path, self.features_file_path,
                self.used_features, self.distance, self.len_feature_weight, self.short_chromatid_ratio_feature_weight,
                self.banding_pattern_feature_weights, self.area_feature_weight, self.initial_neurons_file, self.epochs,
                self.rows, self.cols, self.model_output_file_path, self.dist_matrix_file_path,
                self.generated_karyotype_image_path, self.preckar)

    def __str__(self):
        return "RunInstance:{} [\n\tstart_time={}\n\tend_time={}\n\tinput_image_path={}\n\tfeatures_file_path={}\n\t" \
               "classifier_type={}\n\tused_features={}\n\tdistance={}\n\tlen_feature_weight={}\n\t" \
               "short_chromatid_ratio_feature_weight={}\n\tbanding_pattern_feature_weights={}\n\tarea_feature_weight={}" \
               "\n\tinitial_neurons_file={}\n\tepochs={}\n\trows={}\n\tcols={}\n\tmodel_output_file_path={}\n\t" \
               "dist_matrix_file_path={}\n\tgenerated_karyotype_image_path={}\n\tpreckar={}\n]" \
            .format(self.id, self.start_time, self.end_time, self.input_image_path, self.features_file_path,
                    self.classifier_type, self.used_features, self.distance, self.len_feature_weight,
                    self.short_chromatid_ratio_feature_weight, self.banding_pattern_feature_weights,
                    self.area_feature_weight, self.initial_neurons_file, self.epochs, self.rows, self.cols,
                    self.model_output_file_path, self.dist_matrix_file_path, self.generated_karyotype_image_path,
                    self.preckar)

    def set_end_time(self):
        self.end_time = datetime.strftime(datetime.now(), DB_DATETIME_STR_FORMAT)

    def is_same_model(self, other):
        if self.distance == other.distance:
            if self.distance == WEIGHTED_EUCLIDEAN_DISTANCE:
                return self.classifier_type == other.classifier_type \
                       and self.used_features == other.used_features \
                       and self.len_feature_weight == other.len_feature_weight \
                       and self.short_chromatid_ratio_feature_weight == other.short_chromatid_ratio_feature_weight \
                       and self.banding_pattern_feature_weights == other.banding_pattern_feature_weights \
                       and self.area_feature_weight == other.area_feature_weight \
                       and self.rows == other.rows \
                       and self.cols == other.cols
            else:
                return self.classifier_type == other.classifier_type \
                       and self.used_features == other.used_features \
                       and self.rows == other.rows \
                       and self.cols == other.cols
        return False
