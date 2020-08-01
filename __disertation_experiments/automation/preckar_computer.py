import os

from D_PostProcessSOMResults.accuracy_calculator import compute_accuracy


def main(input_image_path, dist_matrix_file_path, features_file_path):
    preckar = compute_accuracy(pairs_file=os.path.join(input_image_path[:-4], "pairs.txt"),
                               dist_matrix_file=dist_matrix_file_path, features_file=features_file_path)
    print(preckar)


if __name__ == '__main__':
    main(input_image_path=r"f:\GIT\MasterThesisApp\KaryPy\__disertation_experiments\dataset\6.bmp",
         dist_matrix_file_path=r'__disertation_experiments\dataset\6\outputs\features_1\2020-06-02_02-51-34\classifier_dist_matrix.txt',
         features_file_path=r'__disertation_experiments\dataset\6\inputs\features_1.txt')
