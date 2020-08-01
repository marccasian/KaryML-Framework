import numpy as np
from cv2 import cv2
import matplotlib.pyplot as plt
from sympy import symbols, integrate, sqrt, diff, expand

import A_Segmentation.common_operations as common_operations
from C_FeatureExtraction.feature_extractions_constants import *


class Curve:
    def __init__(self, medial_axis_img_path, grade=2):
        self.__img_path = medial_axis_img_path
        self.__polynomial_function_grade = grade
        self.__img = None
        self.threshold_img = None
        self.orig_threshold_img = None
        self.__curve_length = None
        self.__plot_out_img_path = ".".join(self.__img_path.split(".")[:-1]) + "_plot.png"
        self.polynomial_function = None
        self.x_points = list()
        self.y_points = list()

    def get_curve_length(self):
        if self.__curve_length is None:
            self.__compute_curve_length()
        return self.__curve_length

    def __compute_polynomial_function(self):
        if self.__img is None:
            self.__img = common_operations.read_image(self.__img_path)

        ret, self.orig_threshold_img = cv2.threshold(self.__img, 245, 255, cv2.THRESH_BINARY)
        ret, self.threshold_img = cv2.threshold(self.__img, 245, 255, cv2.THRESH_BINARY)
        k = 0
        for i in range(self.threshold_img.shape[0]):
            for j in range(self.threshold_img.shape[1]):
                if all(self.threshold_img[i][j] == 255):
                    self.x_points.append(i)
                    self.y_points.append(j)
                k += 1
        if len(set(self.x_points)) == 1 or len(set(self.y_points)) == 1:
            self.__polynomial_function_grade = 1

        self.polynomial_function = np.poly1d(np.polyfit(self.x_points,
                                                        self.y_points, self.__polynomial_function_grade))

    def get_polynomial_function(self):
        if self.polynomial_function is None:
            self.__compute_polynomial_function()
        return self.polynomial_function

    def __compute_curve_length(self):
        self.get_polynomial_function()
        polynomial_function, t = self.compute_polynomial_function_symbolic()

        self.__curve_length = integrate(
            sqrt(
                1 +
                expand(
                    diff(polynomial_function)
                    *
                    diff(polynomial_function)
                )
            ), (t, 0, self.threshold_img.shape[0])
        ).evalf()
        t = np.linspace(0, self.threshold_img.shape[0], self.threshold_img.shape[0])
        plt.plot(self.x_points, self.y_points, 'o', self.polynomial_function(t), '-')
        plt.savefig(self.__plot_out_img_path)
        plt.close()

    def compute_polynomial_function_symbolic(self):
        if self.polynomial_function is None:
            self.get_polynomial_function()

        t = symbols('t')
        polynomial_function = 0
        coefficients_list = list(self.polynomial_function.coefficients)
        power = len(list(self.polynomial_function.coefficients)) - 1
        for i in range(len(coefficients_list)):
            polynomial_function += coefficients_list[i] * t ** power
            power -= 1

        return polynomial_function, t

    @staticmethod
    def is_valid_curve(curve_path):
        img = common_operations.read_image(curve_path)

        ret, threshold_img = cv2.threshold(img, 245, 255, cv2.THRESH_BINARY)
        k = 0
        x_points, y_points = list(), list()
        for i in range(threshold_img.shape[0]):
            for j in range(threshold_img.shape[1]):
                if all(threshold_img[i][j] == 255):
                    x_points.append(i)
                    y_points.append(j)
                k += 1
        if len(set(x_points)) <= img.shape[0] / VALID_CURVE_POINT_PERCENT \
                or len(set(y_points)) <= img.shape[1] / VALID_CURVE_POINT_PERCENT:
            return False
        return True
