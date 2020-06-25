import os
from typing import List

from D_PostProcessSOMResults.accuracy_calculator import compute_accuracy
from F_Experiments_Helper.db import get_all_runs, update_run_entry
from F_Experiments_Helper.run_instance import RunInstance


def main(input_image_path, dist_matrix_file_path, features_file_path):
    # run_instances: List[RunInstance] = get_all_runs(criteria="WHERE preckar is NULL and end_time is not NULL")
    preckar = compute_accuracy(pairs_file=os.path.join(input_image_path[:-4], "pairs.txt"),
                               dist_matrix_file=dist_matrix_file_path, features_file=features_file_path)
    print(preckar)
    # update_run_entry(i)
    # print(len(run_instances))


if __name__ == '__main__':
    main(input_image_path=r"f:\GIT\MasterThesisApp\KaryPy\__disertation_experiments\dataset\6.bmp",
         dist_matrix_file_path=r'__disertation_experiments\dataset\6\outputs\features_1\2020-06-02_02-51-34\classifier_dist_matrix.txt',
         features_file_path=r'__disertation_experiments\dataset\6\inputs\features_1.txt')
