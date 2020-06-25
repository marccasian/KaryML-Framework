import json
import os
import pickle
import random
import shutil

import F_Experiments_Helper.db as db
import __disertation_experiments.automation.runner as runner
from F_Experiments_Helper.run_instance import RunInstance
from C_FeatureExtraction.feature_extractions_constants import *

OVER_ID = 0

MODELS_LABEL = [
    "E-1",
    "E-2",
    "E-3",
    "WE-3",
    "E-4",
    "WE-5",
    "WE-6",
    "WE-7",
    "E-8",
    "WE-9-0",
    "WE-10-1",
    "WE-11-1",
    "E-12",
    "WE-13-1",
    "E-14",
    "WE-15-3"
]

ORDERED_SELECTED_FEATURES = [
    1,  # Model 4 used features 1 has avg PrecKar 89.1255662622695
    2,  # Model 5 used features 2 has avg PrecKar 86.4099875142842
    3,  # Model 0 used features 3 has avg PrecKar 89.71865671863357
    3,  # Model 8 used features 3 weights 0.3 0.7 has avg PrecKar 88.03329895634404
    4,  # Model 6 used features 4 has avg PrecKar 87.07253639531639
    5,  # Model 9 used features 5 weights 0.3 0.7 has avg PrecKar 87.21327174021701
    6,  # Model 10 used features 6 weights 0.3 0.7 has avg PrecKar 85.66436499517106
    7,  # Model 26 used features 7 weights 0.2 0.2 0.6 has avg PrecKar 87.98756203583098
    8,  # Model 7 used features 8 has avg PrecKar 87.87030055118304
    9,  # Model 12 used features 9 weights 0.6 0.4 has avg PrecKar 89.12659028309345
    10,  # Model 18 used features 10 weights 0.6 0.4 has avg PrecKar 89.40092405029525
    11,  # Model 29 used features 11 weights 0.2 0.6 0.2 has avg PrecKar 89.31775115622007
    12,  # Model 21 used features 12 has avg PrecKar 87.48883786803303
    13,  # Model 33 used features 13 weights 0.1 0.8 0.1 has avg PrecKar 89.40693687939358
    14,  # Model 36 used features 14 has avg PrecKar 87.2078616658087
    15,  # Model 44 used features 15 weights 0.3 0.1 0.3 0.3 has avg PrecKar 89.14354638474389
]

ORDERED_SELECTED_WEIGHTS = [
    None,  # Model 4 used features 1 has avg PrecKar 89.1255662622695
    None,  # Model 5 used features 2 has avg PrecKar 86.4099875142842
    None,  # Model 0 used features 3 has avg PrecKar 89.71865671863357
    [0.3, 0.7],  # Model 8 used features 3 weights 0.3 0.7 has avg PrecKar 88.03329895634404
    None,  # Model 6 used features 4 has avg PrecKar 87.07253639531639
    [0.3, 0.7],  # Model 9 used features 5 weights 0.3 0.7 has avg PrecKar 87.21327174021701
    [0.3, 0.7],  # Model 10 used features 6 weights 0.3 0.7 has avg PrecKar 85.66436499517106
    [0.2, 0.2, 0.6],  # Model 26 used features 7 weights 0.2 0.2 0.6 has avg PrecKar 87.98756203583098
    None,  # Model 7 used features 8 has avg PrecKar 87.87030055118304
    [0.6, 0.4],  # Model 12 used features 9 weights 0.6 0.4 has avg PrecKar 89.12659028309345
    [0.6, 0.4],  # Model 18 used features 10 weights 0.6 0.4 has avg PrecKar 89.40092405029525
    [0.2, 0.6, 0.2],  # Model 29 used features 11 weights 0.2 0.6 0.2 has avg PrecKar 89.31775115622007
    None,  # Model 21 used features 12 weights 0.6 0.4 has avg PrecKar 87.48883786803303
    [0.1, 0.8, 0.1],  # Model 33 used features 13 weights 0.1 0.8 0.1 has avg PrecKar 89.40693687939358
    None,  # Model 36 used features 14 has avg PrecKar 87.2078616658087
    [0.3, 0.1, 0.3, 0.3],  # Model 44 used features 15 weights 0.3 0.1 0.3 0.3 has avg PrecKar 89.14354638474389
]

SELECTED_FEATURES = [
    3,  # Model 0 used features 3 has avg PrecKar 89.71865671863357
    13,  # Model 33 used features 13 weights 0.1 0.8 0.1 has avg PrecKar 89.40693687939358
    10,  # Model 18 used features 10 weights 0.6 0.4 has avg PrecKar 89.40092405029525
    11,  # Model 29 used features 11 weights 0.2 0.6 0.2 has avg PrecKar 89.31775115622007
    15,  # Model 44 used features 15 weights 0.3 0.1 0.3 0.3 has avg PrecKar 89.14354638474389
    9,  # Model 12 used features 9 weights 0.6 0.4 has avg PrecKar 89.12659028309345
    1,  # Model 4 used features 1 has avg PrecKar 89.1255662622695
    3,  # Model 8 used features 3 weights 0.3 0.7 has avg PrecKar 88.03329895634404
    7,  # Model 26 used features 7 weights 0.2 0.2 0.6 has avg PrecKar 87.98756203583098
    8,  # Model 7 used features 8 has avg PrecKar 87.87030055118304
    12,  # Model 21 used features 12 has avg PrecKar 87.48883786803303
    5,  # Model 9 used features 5 weights 0.3 0.7 has avg PrecKar 87.21327174021701
    14,  # Model 36 used features 14 has avg PrecKar 87.2078616658087
    4,  # Model 6 used features 4 has avg PrecKar 87.07253639531639
    2,  # Model 5 used features 2 has avg PrecKar 86.4099875142842
    6,  # Model 10 used features 6 weights 0.3 0.7 has avg PrecKar 85.66436499517106
]

SELECTED_WEIGHTS = [
    None,  # Model 0 used features 3 has avg PrecKar 89.71865671863357
    [0.1, 0.8, 0.1],  # Model 33 used features 13 weights 0.1 0.8 0.1 has avg PrecKar 89.40693687939358
    [0.6, 0.4],  # Model 18 used features 10 weights 0.6 0.4 has avg PrecKar 89.40092405029525
    [0.2, 0.6, 0.2],  # Model 29 used features 11 weights 0.2 0.6 0.2 has avg PrecKar 89.31775115622007
    [0.3, 0.1, 0.3, 0.3],  # Model 44 used features 15 weights 0.3 0.1 0.3 0.3 has avg PrecKar 89.14354638474389
    [0.6, 0.4],  # Model 12 used features 9 weights 0.6 0.4 has avg PrecKar 89.12659028309345
    None,  # Model 4 used features 1 has avg PrecKar 89.1255662622695
    [0.3, 0.7],  # Model 8 used features 3 weights 0.3 0.7 has avg PrecKar 88.03329895634404
    [0.2, 0.2, 0.6],  # Model 26 used features 7 weights 0.2 0.2 0.6 has avg PrecKar 87.98756203583098
    None,  # Model 7 used features 8 has avg PrecKar 87.87030055118304
    None,  # Model 21 used features 12 weights 0.6 0.4 has avg PrecKar 87.48883786803303
    [0.3, 0.7],  # Model 9 used features 5 weights 0.3 0.7 has avg PrecKar 87.21327174021701
    None,  # Model 36 used features 14 has avg PrecKar 87.2078616658087
    None,  # Model 6 used features 4 has avg PrecKar 87.07253639531639
    None,  # Model 5 used features 2 has avg PrecKar 86.4099875142842
    [0.3, 0.7]  # Model 10 used features 6 weights 0.3 0.7 has avg PrecKar 85.66436499517106
]
PRE_RUNS_FEATURES = [
    3,
    5,
    6,
    7,
    1,  # L
    2,  # S
    4,  # B
    8,  # A
    3,  # LS
    5,  # LB
    6,  # SB
    9,  # LA
    9,  # LA
    9,  # LA
    9,  # LA
    9,  # LA
    10,  # SA
    10,  # SA
    10,  # SA
    10,  # SA
    10,  # SA
    12,  # BA
    12,  # BA
    12,  # BA
    12,  # BA
    12,  # BA
    7,  # LSB
    11,  # LSA
    11,  # LSA
    11,  # LSA
    11,  # LSA
    13,  # LBA
    13,  # LBA
    13,  # LBA
    13,  # LBA
    13,  # LBA
    14,  # SBA
    14,  # SBA
    14,  # SBA
    14,  # SBA
    15,  # LSBA
    15,  # LSBA
    15,  # LSBA
    15,  # LSBA
    15,  # LSBA
]

PRE_RUN_WEIGHTS = [
    None, None, None, None,
    None,  # L
    None,  # S
    None,  # B
    None,  # A
    [0.3, 0.7],  # LS
    [0.3, 0.7],  # LB
    [0.3, 0.7],  # SB
    None,  # LA
    [0.6, 0.4],  # LA
    [0.4, 0.6],  # LA
    [0.3, 0.7],  # LA
    [0.7, 0.3],  # LA
    None,  # SA
    [0.4, 0.6],  # SA
    [0.6, 0.4],  # SA
    [0.3, 0.7],  # SA
    [0.7, 0.3],  # SA  - asta
    None,  # BA
    [0.6, 0.4],  # BA
    [0.4, 0.6],  # BA
    [0.7, 0.3],  # BA
    [0.3, 0.7],  # BA
    [0.2, 0.2, 0.6],  # LSB
    None,  # LSA
    [0.2, 0.2, 0.6],  # LSA
    [0.2, 0.6, 0.2],  # LSA
    [0.6, 0.2, 0.2],  # LSA
    None,  # LBA
    [0.15, 0.7, 0.15],  # LBA
    [0.1, 0.8, 0.1],  # LBA
    [0.2, 0.6, 0.2],  # LBA
    [0.3, 0.4, 0.3],  # LBA
    None,  # SBA
    [0.2, 0.2, 0.6],  # SBA
    [0.2, 0.6, 0.2],  # SBA
    [0.6, 0.2, 0.2],  # SBA
    None,  # LSBA
    [0.1, 0.2, 0.6, 0.1],  # LSBA
    [0.2, 0.1, 0.3, 0.2],  # LSBA
    [0.2, 0.2, 0.4, 0.2],  # LSBA
    [0.3, 0.1, 0.3, 0.3],  # LSBA
]

ds_entries = [1, 3, 5, 6, 7, 8]

SOM_CMD_LINE = r"py -3 __disertation_experiments\automation\runner_remainings.py -classifier SOM -ds_entry {} " \
               r"-features {} -epochs 200000 -rows 25 -cols 25"

FSOM_CMD_LINE = r"py -3 __disertation_experiments\automation\runner_remainings.py -classifier FSOM -ds_entry {} " \
                r"-features {} -epochs 200000 -rows 20 -cols 20"

FSOM_DIM = 20

SOM_DIM = 25

DIM = {
    db.SOM_CLASSIFIER_TYPE: 25,
    db.FSOM_CLASSIFIER_TYPE: 20
}


def get_best_initial_input_neurons_file(classifier_type="FSOM"):
    """
    run_instance.classifier_type,
    run_instance.input_image_path,
    run_instance.features_file_path,
    run_instance.used_features,
     run_instance.distance,
     run_instance.rows,
     run_instance.cols
    :param classifier_type:
    :return:
    """
    db.DB_FILE = r'__disertation_experiments\karypy_pre_runs.db'
    feature_indexes = [i for i in range(len(PRE_RUNS_FEATURES))]
    # random.shuffle(ds_entries)
    # random.shuffle(feature_indexes)
    total_models = len(PRE_RUNS_FEATURES) * len(ds_entries)
    ds_descriptor = None
    with open(os.path.join(runner.DS_ROOT_DIR_PATH, "description.json"), "r") as f:
        ds_descriptor = json.load(f)
    print("Searching for best run for {} models".format(total_models))
    dss = runner.ClassifierReqFilesGenerator(ds_descriptor)
    important_msgs = []
    remaining = 0
    best_runs_for_models = []
    new_paths = []
    for i in feature_indexes:
        for ds_entry in ds_entries:
            input_image_path = ds_descriptor[str(ds_entry)][db.INPUT_IMAGE_PATH_COLUMN_NAME]
            root_dir_path, features_file_path, features_weight_file_path, feature_used_list, outputs_dir_path = dss.generate_input_feature_files(
                str(ds_entry), PRE_RUNS_FEATURES[i], PRE_RUN_WEIGHTS[i])
            '''
            used_features,
            input_image_path,
            features_file_path,
            len_feature_weight,
            short_chromatid_ratio_feature_weight, 
            banding_pattern_feature_weights,
            area_feature_weight, distance
            '''
            a = RunInstance(classifier_type=classifier_type, input_image_path=input_image_path,
                            features_file_path=features_file_path, used_features=PRE_RUNS_FEATURES[i],
                            distance=runner.determine_distance_type(False, PRE_RUN_WEIGHTS[i]), epochs=200000,
                            rows=DIM[classifier_type], cols=DIM[classifier_type],
                            len_feature_weight=runner.get_feature_weight(feature_used_list, CHROMOSOME_LEN_KEY),
                            short_chromatid_ratio_feature_weight=runner.get_feature_weight(feature_used_list,
                                                                                           SHORT_CHROMATID_RATIO_KEY),
                            banding_pattern_feature_weights=runner.get_feature_weight(feature_used_list,
                                                                                      BANDAGE_PROFILE_KEY),
                            area_feature_weight=runner.get_feature_weight(feature_used_list, CHROMOSOME_AREA_KEY))
            runs = db.get_similar_runs(a, over_id=OVER_ID)

            if len(runs) > 0:
                if runs[0].preckar < 87.5:
                    if len(runs) < 10:
                        msg = "WAAAAAAAAAAAAAAAAARRRRNING: {} not enough runs found. Just {}".format("Model: {}".format(
                            "{}-{}-{}".format(classifier_type, str(ds_entry), PRE_RUNS_FEATURES[i])), format(len(runs)))
                        remaining += 10 - len(runs)
                        important_msgs.append(msg)
                if PRE_RUN_WEIGHTS[i]:
                    print("Model: {} best_initial_neurons={} preckar={}".format(
                        "{}-{}-{}-{}".format(classifier_type, str(ds_entry), PRE_RUNS_FEATURES[i],
                                             "_".join([str(k) for k in PRE_RUN_WEIGHTS[i]])),
                        runs[0].initial_neurons_file,
                        runs[0].preckar))
                    pass
                else:
                    print("Model: {} best_initial_neurons={} preckar={}".format(
                        "{}-{}-{}".format(classifier_type, str(ds_entry), PRE_RUNS_FEATURES[i]),
                        runs[0].initial_neurons_file,
                        runs[0].preckar))
                    pass

                if not os.path.exists(runs[0].initial_neurons_file):
                    runs[0].initial_neurons_file = runs[0].initial_neurons_file.replace("outputs",
                                                                                        "{}_needed_outs".format(
                                                                                            classifier_type.lower()))

                if not os.path.exists(runs[0].initial_neurons_file):
                    print("Not found {}".format(runs[0].initial_neurons_file))
                else:
                    best_runs_for_models.append(runs[0])
                    new_path = runs[0].initial_neurons_file.replace("outputs",
                                                                    "{}_needed_outs".format(classifier_type.lower()))
                    new_path_dir, file_name = os.path.split(new_path)
                    new_paths.append((new_path_dir, file_name, new_path))
            else:
                msg = "WAAAAAAAAAAAAAAAAARRRRNING: {} no funs found".format("Model: {}".format(
                    "{}-{}-{}".format(classifier_type, str(ds_entry), PRE_RUNS_FEATURES[i])))
                important_msgs.append(msg)
    for i in important_msgs:
        print(i)

    # for i in new_paths:
    #     print(i)

    print("ar mai trebui rulate {}".format(remaining))

    return best_runs_for_models


def decide_models_to_evaluate(classifier_type="FSOM"):
    best_runs = get_best_initial_input_neurons_file(classifier_type)
    runs_per_models = dict()
    averaged_models_for_all_ds_entries = dict()
    # for i in ds_entries:
    #     per_ds_entry[i] = list()
    #     averaged_models_for_all_ds_entries[i] = {db.EUCLIDEAN_DISTANCE: list()}
    #     averaged_models_for_all_ds_entries[i] = {db.WEIGHTED_EUCLIDEAN_DISTANCE: list()}
    # for run in best_runs:
    #     per_ds_entry[int(run.input_image_path[-5:][:1])].append(run)
    feature_indexes = [i for i in range(len(PRE_RUNS_FEATURES))]
    prerun_precar_vals = list()
    for i in feature_indexes:
        current_preckars = list()
        runs_per_models[i] = list()
        used_features = PRE_RUNS_FEATURES[i]
        len_w = None
        sh_w = None
        ban_w = None
        are_w = None
        c_distance = db.EUCLIDEAN_DISTANCE

        if PRE_RUN_WEIGHTS[i]:
            idx = 0
            c_distance = db.WEIGHTED_EUCLIDEAN_DISTANCE
            if used_features & 0x01:
                len_w = PRE_RUN_WEIGHTS[i][idx]
                idx += 1
            if used_features & 0x02:
                sh_w = PRE_RUN_WEIGHTS[i][idx]
                idx += 1
            if used_features & 0x04:
                ban_w = PRE_RUN_WEIGHTS[i][idx]
                idx += 1
            if used_features & 0x08:
                are_w = PRE_RUN_WEIGHTS[i][idx]
                idx += 1
        a = RunInstance(classifier_type=classifier_type, input_image_path=None, features_file_path=None,
                        used_features=used_features, distance=c_distance, epochs=None, rows=DIM[classifier_type],
                        cols=DIM[classifier_type], len_feature_weight=len_w, short_chromatid_ratio_feature_weight=sh_w,
                        banding_pattern_feature_weights=ban_w, area_feature_weight=are_w)
        suma = 0
        for r in best_runs:
            if r.is_same_model(a):
                runs_per_models[i].append(r)
                suma += r.preckar
                current_preckars.append(r.preckar)
        prerun_precar_vals.append((PRE_RUNS_FEATURES[i], PRE_RUN_WEIGHTS[i], current_preckars))
        averaged_models_for_all_ds_entries[i] = suma / len(runs_per_models[i])

    ordered_dict = {k: v for k, v in
                    sorted(averaged_models_for_all_ds_entries.items(), key=lambda item: item[1], reverse=True)}
    for k in ordered_dict:
        if PRE_RUN_WEIGHTS[k]:
            print("Model {} used features {} weights {} has avg PrecKar {} ".format(k, PRE_RUNS_FEATURES[k],
                                                                                    " ".join(
                                                                                        [str(o) for o in
                                                                                         PRE_RUN_WEIGHTS[k]]),
                                                                                    ordered_dict[k]))
        else:
            print("Model {} used features {} has avg PrecKar {} ".format(k, PRE_RUNS_FEATURES[k], ordered_dict[k]))
    with open("{}_preruns_preckar_vals_obj.pyobj".format(classifier_type), "wb") as pf:
        pickle.dump(prerun_precar_vals, pf)
    return averaged_models_for_all_ds_entries


if __name__ == '__main__':
    decide_models_to_evaluate("SOM")
