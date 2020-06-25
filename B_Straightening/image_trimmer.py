import os

from cv2 import cv2

import B_Straightening.compute_projection_vector as compute_projection_vector
import A_Segmentation.common_operations as common_operations
# todo ... remove noise.... threshold maybe


class TrimImage:
    def __init__(self):
        self.compute_projection_vector_obj = compute_projection_vector.ComputeProjectionVector()

    def trim(self, image_path):
        x1, x2, y1, y2 = self.__get_cut_points(image_path)
        if self.__validate_points(x1, x2, y1, y2):
            self.__trim_and_save_image(image_path, x1, x2, y1, y2)

    def __get_cut_points(self, image_path):
        h_vector = self.compute_projection_vector_obj.get_horizontal_projection_vector(image_path)
        v_vector = self.compute_projection_vector_obj.get_vertical_projection_vector(image_path)
        x1, x2, y1, y2 = None, None, None, None
        for y1 in range(len(h_vector)):
            if h_vector[y1] > 0:
                break
        for y2 in range(len(h_vector) - 1, -1, -1):
            if h_vector[y2] > 0:
                break
        for x1 in range(len(v_vector)):
            if v_vector[x1] > 0:
                break
        for x2 in range(len(v_vector) - 1, -1, -1):
            if v_vector[x2] > 0:
                break
        return x1, x2, y1, y2

    @staticmethod
    def __validate_points(x1, x2, y1, y2):
        return x1 < x2 and y1 < y2

    @staticmethod
    def __trim_and_save_image(image_path, x1, x2, y1, y2):
        image = common_operations.read_image(image_path)
        new_image = image[y1:y2, x1:x2]
        cv2.imwrite(image_path, new_image)


if __name__ == '__main__':
    path = '1.jpg'
    obj = TrimImage()
    obj.trim(path)
