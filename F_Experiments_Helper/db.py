import os
import sqlite3
import sys
import traceback
from sqlite3 import Error
from F_Experiments_Helper.run_instance import RunInstance
from a_Common.constants import *

DB_FILE = r"__disertation_experiments\karypy_runs.db"

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
INITIAL_NEURONS_FILE_COLUMN_NAME = "initial_neurons_file"  # initial model configuration (either existing one, meaning
# that the model starts from a given config, either the initial neurons of the model will be saved
EPOCHS_COLUMN_NAME = "epochs"  # number of training epochs for the model
ROWS_COLUMN_NAME = "rows"
COLS_COLUMN_NAME = "cols"

# outs
MODEL_OUTPUT_FILE_PATH_COLUMN_NAME = "model_output_file_path"  # som output file
DIST_MATRIX_FILE_PATH_COLUMN_NAME = "dist_matrix_file_path"
GENERATED_KARYOTYPE_IMAGE_PATH_COLUMN_NAME = "generated_karyotype_image_path"
PRECKAR_COLUMN_NAME = "preckar"

# metadata columns
ID_COLUMN_INDEX = 0
START_TIME_COLUMN_INDEX = 1
END_TIME_COLUMN_INDEX = 2

# input columns
CLASSIFIER_TYPE_COLUMN_INDEX = 3
INPUT_IMAGE_PATH_COLUMN_INDEX = 4
FEATURES_FILE_PATH_COLUMN_INDEX = 5
USED_FEATURES_COLUMN_INDEX = 6
DISTANCE_COLUMN_INDEX = 7
LEN_FEATURE_WEIGHT_COLUMN_INDEX = 8
SHORT_CHROMATID_RATIO_FEATURE_WEIGHT_COLUMN_INDEX = 9
BANDING_PATTERN_FEATURE_WEIGHTS_COLUMN_INDEX = 10
AREA_FEATURE_WEIGHT_COLUMN_INDEX = 11
INITIAL_NEURONS_FILE_COLUMN_INDEX = 12
EPOCHS_COLUMN_INDEX = 13
ROWS_COLUMN_INDEX = 14
COLS_COLUMN_INDEX = 15

# outs
MODEL_OUTPUT_FILE_PATH_COLUMN_INDEX = 16
DIST_MATRIX_FILE_PATH_COLUMN_INDEX = 17
GENERATED_KARYOTYPE_IMAGE_PATH_COLUMN_INDEX = 18
PRECKAR_COLUMN_INDEX = 19


def create_connection(db_file):
    """ create a database connection to the SQLite database
        specified by the db_file
    :param db_file: database file
    :return: Connection object or None
    """
    conn = None
    try:
        conn = sqlite3.connect(db_file)
    except Error as e:
        print(e)

    return conn


def create_runs_table():
    """
    table run:
        # meta info
        id - autoincrement
        start_time - text
        end_time - text

        # inputs
        classifier_type - text (SOM  or FSOM atm)
        input_image_path - text
        features_file_path - text
        used_features - integer (0x01 len, 0x02-centromere, 0x04 banding, 0x08 area)
        distance - text (euclidean, weighted-euclidean, manhattan)
        1_weight - real # weights for len feature
        2_weight - real # weights centromere feature
        4_weight - real # weights banding pattern feature
        8_weight - real # weights for area feature
        used_neurons_file - text (if initial model configuration was loaded from a file
        epochs - integer (number of training epochs for the model)
        rows - integer
        cols - integer

        # outs
        model_output_file - text (som output file)
        dist_matrix_file_path - text
        generated_karyotype_image_path - text
        preckar - real

    :return:
    """
    conn = create_connection(DB_FILE)
    with conn:
        conn.execute('''
            CREATE TABLE run ({} INTEGER PRIMARY KEY, {} TEXT , {} TEXT, {} TEXT, {} TEXT, {} TEXT, {} INTEGER, {} TEXT, 
            {} REAL, {} REAL, {} TEXT, {} REAL, {} TEXT, {} INTEGER, {} INTEGER, {} INTEGER, {} TEXT, {} TEXT, {} TEXT, 
            {} REAL)
        '''.format(ID_COLUMN_NAME, START_TIME_COLUMN_NAME, END_TIME_COLUMN_NAME, CLASSIFIER_TYPE_COLUMN_NAME,
                   INPUT_IMAGE_PATH_COLUMN_NAME, FEATURES_FILE_PATH_COLUMN_NAME, USED_FEATURES_COLUMN_NAME,
                   DISTANCE_COLUMN_NAME, LEN_FEATURE_WEIGHT_COLUMN_NAME,
                   SHORT_CHROMATID_RATIO_FEATURE_WEIGHT_COLUMN_NAME, BANDING_PATTERN_FEATURE_WEIGHTS_COLUMN_NAME,
                   AREA_FEATURE_WEIGHT_COLUMN_NAME, INITIAL_NEURONS_FILE_COLUMN_NAME, EPOCHS_COLUMN_NAME,
                   ROWS_COLUMN_NAME, COLS_COLUMN_NAME, MODEL_OUTPUT_FILE_PATH_COLUMN_NAME,
                   DIST_MATRIX_FILE_PATH_COLUMN_NAME, GENERATED_KARYOTYPE_IMAGE_PATH_COLUMN_NAME, PRECKAR_COLUMN_NAME))


def insert_new_run(run_instance: RunInstance):
    conn = create_connection(DB_FILE)
    with conn:
        sql = ''' INSERT INTO run({},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{})
                     VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?) ''' \
            .format(START_TIME_COLUMN_NAME, END_TIME_COLUMN_NAME, CLASSIFIER_TYPE_COLUMN_NAME,
                    INPUT_IMAGE_PATH_COLUMN_NAME, FEATURES_FILE_PATH_COLUMN_NAME, USED_FEATURES_COLUMN_NAME,
                    DISTANCE_COLUMN_NAME, LEN_FEATURE_WEIGHT_COLUMN_NAME,
                    SHORT_CHROMATID_RATIO_FEATURE_WEIGHT_COLUMN_NAME, BANDING_PATTERN_FEATURE_WEIGHTS_COLUMN_NAME,
                    AREA_FEATURE_WEIGHT_COLUMN_NAME, INITIAL_NEURONS_FILE_COLUMN_NAME, EPOCHS_COLUMN_NAME,
                    ROWS_COLUMN_NAME, COLS_COLUMN_NAME, MODEL_OUTPUT_FILE_PATH_COLUMN_NAME,
                    DIST_MATRIX_FILE_PATH_COLUMN_NAME, GENERATED_KARYOTYPE_IMAGE_PATH_COLUMN_NAME, PRECKAR_COLUMN_NAME)
        cur = conn.cursor()
        cur.execute(sql, run_instance.get_insert_tuple())
        inserted_row_id = cur.lastrowid
        if inserted_row_id is None:
            raise ValueError("Insert Failed. Query: {}. Values: {}".format(sql, run_instance.get_insert_tuple()))
    return inserted_row_id


def delete_run(run_instance: RunInstance):
    conn = create_connection(DB_FILE)
    with conn:
        sql = ''' DELETE FROM run WHERE {} = {}'''.format(ID_COLUMN_NAME, run_instance.id)
        cur = conn.cursor()
        cur.execute(sql)
        inserted_row_id = cur.lastrowid
        if inserted_row_id is None:
            raise ValueError("Insert Failed. Query: {}. Values: {}".format(sql, run_instance.get_insert_tuple()))
    return inserted_row_id


def update_run_entry(run_instance: RunInstance):
    conn = create_connection(DB_FILE)
    with conn:
        sql = ''' UPDATE run SET {} = ?, {} = ? , {} = ? , {} = ? ,{} = ? ,{} = ? ,{} = ? ,{} = ? ,{} = ? ,{} = ? ,{} = ? ,{} = ?, {} = ? ,{} = ? ,{} = ? ,{} = ? ,{} = ? ,{} = ? ,{} = ? WHERE {} = ?'''. \
            format(START_TIME_COLUMN_NAME, END_TIME_COLUMN_NAME, CLASSIFIER_TYPE_COLUMN_NAME,
                   INPUT_IMAGE_PATH_COLUMN_NAME, FEATURES_FILE_PATH_COLUMN_NAME,
                   USED_FEATURES_COLUMN_NAME, DISTANCE_COLUMN_NAME,
                   LEN_FEATURE_WEIGHT_COLUMN_NAME,
                   SHORT_CHROMATID_RATIO_FEATURE_WEIGHT_COLUMN_NAME,
                   BANDING_PATTERN_FEATURE_WEIGHTS_COLUMN_NAME,
                   AREA_FEATURE_WEIGHT_COLUMN_NAME, INITIAL_NEURONS_FILE_COLUMN_NAME,
                   EPOCHS_COLUMN_NAME, ROWS_COLUMN_NAME, COLS_COLUMN_NAME,
                   MODEL_OUTPUT_FILE_PATH_COLUMN_NAME, DIST_MATRIX_FILE_PATH_COLUMN_NAME,
                   GENERATED_KARYOTYPE_IMAGE_PATH_COLUMN_NAME, PRECKAR_COLUMN_NAME,
                   ID_COLUMN_NAME)
        conn.execute(sql, run_instance.get_update_tuple())
        conn.commit()


def get_all_runs(what="*", criteria=""):
    conn = create_connection(DB_FILE)
    runs = list()
    with conn:
        cur = conn.cursor()
        sql = "SELECT {} from run {};".format(what, criteria)
        cur.execute(sql)
        for i in cur.fetchall():
            if what == "*":
                runs.append(get_run_instance_obj_from_db_entry(i))
            else:
                runs.append(i)
    return runs


def get_run_instance_obj_from_db_entry(i):
    return RunInstance(
        id=i[ID_COLUMN_INDEX], start_time=i[START_TIME_COLUMN_INDEX], end_time=i[END_TIME_COLUMN_INDEX],
        classifier_type=i[CLASSIFIER_TYPE_COLUMN_INDEX], input_image_path=i[INPUT_IMAGE_PATH_COLUMN_INDEX],
        features_file_path=i[FEATURES_FILE_PATH_COLUMN_INDEX], used_features=i[USED_FEATURES_COLUMN_INDEX],
        distance=i[DISTANCE_COLUMN_INDEX], len_feature_weight=i[LEN_FEATURE_WEIGHT_COLUMN_INDEX],
        short_chromatid_ratio_feature_weight=i[SHORT_CHROMATID_RATIO_FEATURE_WEIGHT_COLUMN_INDEX],
        banding_pattern_feature_weights=float(i[BANDING_PATTERN_FEATURE_WEIGHTS_COLUMN_INDEX]) if i[
            BANDING_PATTERN_FEATURE_WEIGHTS_COLUMN_INDEX] else None,
        area_feature_weight=i[AREA_FEATURE_WEIGHT_COLUMN_INDEX],
        initial_neurons_file=i[INITIAL_NEURONS_FILE_COLUMN_INDEX], epochs=i[EPOCHS_COLUMN_INDEX],
        rows=i[ROWS_COLUMN_INDEX],
        cols=i[COLS_COLUMN_INDEX], model_output_file_path=i[MODEL_OUTPUT_FILE_PATH_COLUMN_INDEX],
        dist_matrix_file_path=i[DIST_MATRIX_FILE_PATH_COLUMN_INDEX],
        generated_karyotype_image_path=i[GENERATED_KARYOTYPE_IMAGE_PATH_COLUMN_INDEX], preckar=i[PRECKAR_COLUMN_INDEX])


def get_best_run_so_far_with_similar_input(run_instance: RunInstance, logger=None):
    conn = create_connection(DB_FILE)
    run = None
    with conn:
        # conn.set_trace_callback(print)
        try:
            cur = conn.cursor()
            sql = "SELECT * FROM run WHERE {} = ? and {} = ? and {} = ? and {} = ? and {} = ? and {} = ? and {} = ? ".format(
                CLASSIFIER_TYPE_COLUMN_NAME, INPUT_IMAGE_PATH_COLUMN_NAME,
                FEATURES_FILE_PATH_COLUMN_NAME, USED_FEATURES_COLUMN_NAME, DISTANCE_COLUMN_NAME, ROWS_COLUMN_NAME,
                COLS_COLUMN_NAME)
            bindings = [run_instance.classifier_type, run_instance.input_image_path,
                        run_instance.features_file_path,
                        run_instance.used_features, run_instance.distance, run_instance.rows, run_instance.cols]
            if run_instance.len_feature_weight:
                sql += " AND {} = ? ".format(LEN_FEATURE_WEIGHT_COLUMN_NAME)
                bindings.append(run_instance.len_feature_weight)
            else:
                sql += " AND {} IS NULL ".format(LEN_FEATURE_WEIGHT_COLUMN_NAME)

            if run_instance.short_chromatid_ratio_feature_weight:
                sql += " AND {} = ? ".format(SHORT_CHROMATID_RATIO_FEATURE_WEIGHT_COLUMN_NAME)
                bindings.append(run_instance.short_chromatid_ratio_feature_weight)
            else:
                sql += " AND {} IS NULL ".format(SHORT_CHROMATID_RATIO_FEATURE_WEIGHT_COLUMN_NAME)
            if run_instance.banding_pattern_feature_weights:
                sql += " AND {} = ? ".format(BANDING_PATTERN_FEATURE_WEIGHTS_COLUMN_NAME)
                bindings.append(run_instance.banding_pattern_feature_weights)
            else:
                sql += " AND {} IS NULL ".format(BANDING_PATTERN_FEATURE_WEIGHTS_COLUMN_NAME)
            if run_instance.area_feature_weight:
                sql += " AND {} = ? ".format(AREA_FEATURE_WEIGHT_COLUMN_NAME)
                bindings.append(run_instance.area_feature_weight)
            else:
                sql += " AND {} IS NULL ".format(AREA_FEATURE_WEIGHT_COLUMN_NAME)

            sql += " AND {} is not NULL ORDER BY {} DESC LIMIT 1;".format(PRECKAR_COLUMN_NAME, PRECKAR_COLUMN_NAME)

            cur.execute(sql, tuple(bindings))
            i = cur.fetchone()
            if i:
                run = get_run_instance_obj_from_db_entry(i)
        except:
            msg = "Exception while searching for best_run_so_far_with_similar_input. Traceback: {}".format(
                traceback.format_exc())
            print(msg)
            if logger:
                logger.warning(msg)
        # conn.set_trace_callback(None)
    return run


def get_similar_runs(run_instance: RunInstance, over_id=None):
    conn = create_connection(DB_FILE)
    runs = list()
    with conn:
        # conn.set_trace_callback(print)
        try:
            cur = conn.cursor()
            sql = "SELECT * FROM run WHERE {} = ? and {} = ? and {} = ? and {} = ? and {} = ? and {} = ? and {} = ? ".format(
                CLASSIFIER_TYPE_COLUMN_NAME, INPUT_IMAGE_PATH_COLUMN_NAME, FEATURES_FILE_PATH_COLUMN_NAME,
                USED_FEATURES_COLUMN_NAME, DISTANCE_COLUMN_NAME, ROWS_COLUMN_NAME, COLS_COLUMN_NAME)
            bindings = [run_instance.classifier_type, run_instance.input_image_path, run_instance.features_file_path,
                        run_instance.used_features, run_instance.distance, run_instance.rows, run_instance.cols]
            if over_id:
                sql += " AND {} >= ? ".format(ID_COLUMN_NAME)
                bindings.append(over_id)
            if run_instance.len_feature_weight:
                sql += " AND {} = ? ".format(LEN_FEATURE_WEIGHT_COLUMN_NAME)
                bindings.append(run_instance.len_feature_weight)
            else:
                sql += " AND {} IS NULL ".format(LEN_FEATURE_WEIGHT_COLUMN_NAME)

            if run_instance.short_chromatid_ratio_feature_weight:
                sql += " AND {} = ? ".format(SHORT_CHROMATID_RATIO_FEATURE_WEIGHT_COLUMN_NAME)
                bindings.append(run_instance.short_chromatid_ratio_feature_weight)
            else:
                sql += " AND {} IS NULL ".format(SHORT_CHROMATID_RATIO_FEATURE_WEIGHT_COLUMN_NAME)
            if run_instance.banding_pattern_feature_weights:
                sql += " AND {} = ? ".format(BANDING_PATTERN_FEATURE_WEIGHTS_COLUMN_NAME)
                bindings.append(run_instance.banding_pattern_feature_weights)
            else:
                sql += " AND {} IS NULL ".format(BANDING_PATTERN_FEATURE_WEIGHTS_COLUMN_NAME)
            if run_instance.area_feature_weight:
                sql += " AND {} = ? ".format(AREA_FEATURE_WEIGHT_COLUMN_NAME)
                bindings.append(run_instance.area_feature_weight)
            else:
                sql += " AND {} IS NULL ".format(AREA_FEATURE_WEIGHT_COLUMN_NAME)

            sql += " AND {} is not NULL ORDER BY {} desc;".format(PRECKAR_COLUMN_NAME, PRECKAR_COLUMN_NAME)

            cur.execute(sql, tuple(bindings))
            for i in cur.fetchall():
                runs.append(get_run_instance_obj_from_db_entry(i))
        except:
            msg = "Exception while searching for best_run_so_far_with_similar_input. Traceback: {}".format(
                traceback.format_exc())
            print(msg)
        # conn.set_trace_callback(None)
    return runs


def how_many_similar_input_runs_so_far(run_instance: RunInstance, logger=None, over_id=None):
    conn = create_connection(DB_FILE)
    run = None
    with conn:
        # conn.set_trace_callback(print)
        try:
            cur = conn.cursor()
            sql = "SELECT count(*) FROM run WHERE {} = ? and {} = ? and {} = ? and {} = ? and {} = ? and {} = ? and {} = ? ".format(
                CLASSIFIER_TYPE_COLUMN_NAME, INPUT_IMAGE_PATH_COLUMN_NAME,
                FEATURES_FILE_PATH_COLUMN_NAME, USED_FEATURES_COLUMN_NAME, DISTANCE_COLUMN_NAME, ROWS_COLUMN_NAME,
                COLS_COLUMN_NAME)
            bindings = [run_instance.classifier_type, run_instance.input_image_path,
                        run_instance.features_file_path, run_instance.used_features, run_instance.distance,
                        run_instance.rows, run_instance.cols]
            if over_id:
                sql += " AND {} >= ?".format(ID_COLUMN_NAME)
                bindings.append(over_id)
            if run_instance.len_feature_weight:
                sql += " AND {} = ? ".format(LEN_FEATURE_WEIGHT_COLUMN_NAME)
                bindings.append(run_instance.len_feature_weight)
            else:
                sql += " AND {} IS NULL ".format(LEN_FEATURE_WEIGHT_COLUMN_NAME)

            if run_instance.short_chromatid_ratio_feature_weight:
                sql += " AND {} = ? ".format(SHORT_CHROMATID_RATIO_FEATURE_WEIGHT_COLUMN_NAME)
                bindings.append(run_instance.short_chromatid_ratio_feature_weight)
            else:
                sql += " AND {} IS NULL ".format(SHORT_CHROMATID_RATIO_FEATURE_WEIGHT_COLUMN_NAME)
            if run_instance.banding_pattern_feature_weights:
                sql += " AND {} = ? ".format(BANDING_PATTERN_FEATURE_WEIGHTS_COLUMN_NAME)
                bindings.append(run_instance.banding_pattern_feature_weights)
            else:
                sql += " AND {} IS NULL ".format(BANDING_PATTERN_FEATURE_WEIGHTS_COLUMN_NAME)
            if run_instance.area_feature_weight:
                sql += " AND {} = ? ".format(AREA_FEATURE_WEIGHT_COLUMN_NAME)
                bindings.append(run_instance.area_feature_weight)
            else:
                sql += " AND {} IS NULL ".format(AREA_FEATURE_WEIGHT_COLUMN_NAME)

            sql += " AND {} is not NULL ORDER BY {} DESC LIMIT 1;".format(PRECKAR_COLUMN_NAME, PRECKAR_COLUMN_NAME)

            cur.execute(sql, tuple(bindings))
            i = cur.fetchone()[0]
            return i
        except:
            msg = "Exception while searching for best_run_so_far_with_similar_input. Traceback: {}".format(
                traceback.format_exc())
            print(msg)
            if logger:
                logger.warning(msg)
            # conn.set_trace_callback(None)
            return 0


def test_get_best_similar_run():
    '''
    6	93.843548674743	15	__disertation_experiments\dataset\5\5.bmp	__disertation_experiments\dataset\5\inputs\features_15.txt	0.3	0.1	0.3	0.3	FSOM	Weighted Euclidean
    :return:
    '''
    rr = RunInstance(
        classifier_type=FSOM_CLASSIFIER_TYPE,
        input_image_path=r"__disertation_experiments\dataset\5\5.bmp",
        features_file_path=r"__disertation_experiments\dataset\5\inputs\features_15.txt",
        used_features=15,
        distance=WEIGHTED_EUCLIDEAN_DISTANCE,
        len_feature_weight=0.3,
        short_chromatid_ratio_feature_weight=0.1,
        banding_pattern_feature_weights=0.3,
        area_feature_weight=0.3,
        rows=20,
        cols=20, epochs=200000
    )
    print(rr)
    print("====================================")
    print(get_best_run_so_far_with_similar_input(rr, None))
    print(how_many_similar_input_runs_so_far(rr, None, 2937))


def test_main_flow():
    global DB_FILE
    DB_FILE = DB_FILE + "_test"

    create_runs_table()
    r = RunInstance(end_time=None, classifier_type="SOM", input_image_path="not_exists_i_path",
                    features_file_path="not existing f path", used_features=0x0f, distance=MANHATTAN_DISTANCE,
                    len_feature_weight=None, short_chromatid_ratio_feature_weight=None,
                    banding_pattern_feature_weights=None, area_feature_weight=None, initial_neurons_file=None,
                    epochs=200000, rows=50, cols=50)
    print(r)
    print("=========================")
    r.id = insert_new_run(r)
    print(r)
    r.set_end_time()
    update_run_entry(r)
    print("=========================de aici")
    for i in get_all_runs():
        print(str(i))
    delete_run(r)
    print("=========================")
    print(get_all_runs())
    os.remove(DB_FILE)


if __name__ == '__main__':
    create_runs_table()
