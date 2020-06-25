import json
import os

DS_ENTRIES = [1, 3, 5, 6, 7, 8]

from F_Experiments_Helper.db import INPUT_IMAGE_PATH_COLUMN_NAME, FEATURES_FILE_PATH_COLUMN_NAME

DS_ROOT_DIR_PATH = r'__disertation_experiments\dataset'
DS_ENTRY_DIR_WORKING_DIR_ROOT_PATH = "root_dir_path"


def main():
    """
    I need a json having following format:
    {
        DS_ENTRY_NR: {
           FEATURES_FILE_PATH_COLUMN_NAME: str,
           INPUT_IMAGE_PATH_COLUMN_NAME: str,
           DS_ENTRY_DIR_WORKING_DIR_ROOT_PATH: str,
        },
        ...
    }
    :return:
    """
    ds_description_json = dict()
    for i in DS_ENTRIES:
        ds_entry_root_dir_name = str(i)
        ds_entry_root_dir_path = os.path.join(DS_ROOT_DIR_PATH, ds_entry_root_dir_name)
        ds_entry_working_dir_path = os.path.join(ds_entry_root_dir_path, ds_entry_root_dir_name)
        input_image_path = os.path.join(ds_entry_root_dir_path, "{}.bmp".format(i))
        features_file_path = os.path.join(ds_entry_root_dir_path, "straight_features.json")
        ds_description_json[i] = {
            FEATURES_FILE_PATH_COLUMN_NAME: features_file_path,
            INPUT_IMAGE_PATH_COLUMN_NAME: input_image_path,
            DS_ENTRY_DIR_WORKING_DIR_ROOT_PATH: ds_entry_working_dir_path
        }
    with open(os.path.join(DS_ROOT_DIR_PATH, "description.json"), "w") as f:
        json.dump(ds_description_json, f, indent=4)


if __name__ == '__main__':
    main()
