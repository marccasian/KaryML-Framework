from cv2 import cv2

import A_Segmentation.common_operations as common_operations


class ComputeProjectionVector:
    def __init__(self):
        self.img_path = None

    def get_horizontal_projection_vector(self, image_path):
        self.img_path = image_path
        image = common_operations.read_image(self.img_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ret, img1 = cv2.threshold(gray, 245, 255, cv2.THRESH_BINARY_INV)
        h_projection_vector = list()
        for i in range(img1.shape[0]):
            value = 0
            for j in range(img1.shape[1]):
                if img1[i][j] == 0:
                    continue
                value += 1
            h_projection_vector.append(value)
        return h_projection_vector

    def get_vertical_projection_vector(self, image_path):
        self.img_path = image_path
        image = common_operations.read_image(self.img_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ret, img1 = cv2.threshold(gray, 245, 255, cv2.THRESH_BINARY_INV)
        v_projection_vector = list()
        for i in range(img1.shape[1]):
            value = 0
            for j in range(img1.shape[0]):
                if img1[j][i] == 0:
                    continue
                value += 1
            v_projection_vector.append(value)
        return v_projection_vector
