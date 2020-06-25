import os
from typing import List

from D_PostProcessSOMResults.accuracy_calculator import compute_accuracy
from F_Experiments_Helper.db import get_all_runs, update_run_entry
from F_Experiments_Helper.run_instance import RunInstance


def main():
    run_instances: List[RunInstance] = get_all_runs(criteria="WHERE preckar is NULL and end_time is not NULL")
    for i in run_instances:
        print(i)
        preckar = compute_accuracy(pairs_file=os.path.join(i.input_image_path[:-4], "pairs.txt"),
                                   dist_matrix_file=i.dist_matrix_file_path, features_file=i.features_file_path,
                                   neurons_file=i.initial_neurons_file, timestamp_str=i.start_time)
        i.preckar = preckar
        update_run_entry(i)
    print(len(run_instances))


if __name__ == '__main__':
    main()
