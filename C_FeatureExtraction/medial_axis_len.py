from scipy import ndimage

import numpy as np
from cv2 import cv2
import matplotlib.pyplot as plt
from sympy import symbols, integrate, sqrt, diff, evalf, expand
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

        # print("Polynomus= " + str(polynomial_function))

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

        # print("shape= " + str(img.shape))
        # print("lungime= " + str(curve_length))
        # plot medial axis and best matching polynomial function
        t = np.linspace(0, self.threshold_img.shape[0], self.threshold_img.shape[0])  # (x), len(x))
        my_plot = plt.plot(self.x_points, self.y_points, 'o', self.polynomial_function(t), '-')
        # rotated_plot = ndimage.rotate(my_plot, 90)
        # import png
        # png.fromarray(rotated_plot).save(self.__plot_out_img_path+"_rotated.png")
        plt.savefig(self.__plot_out_img_path)
        plt.close()
        # plt.show()

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

        ret, orig_threshold_img = cv2.threshold(img, 245, 255, cv2.THRESH_BINARY)
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


if __name__ == "__main__":
    medial_path = r'd:\GIT\Karyotyping-Project\PythonProject\Z_Images\brut1-1\Outputs\35-145\ShortChromatidRatioHelperDir\ChromosomeLenHelperDir\35-145-0\C_Feature_Extractor\MedialAxis\35-145-0-final.bmp'
    obj = Curve(medial_path, grade=2)
    leng = obj.get_curve_length()
    print(leng)
    print(obj.x_points)
    print(set(obj.x_points))
    print(obj.y_points)
    print(set(obj.y_points))
    print(obj.polynomial_function)
