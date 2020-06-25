import os

import sys

import imutils
from cv2 import cv2
import pickle
from mpmath import linspace
import sympy as sp
import numpy as np
import math
from C_FeatureExtraction.medial_axis import MedialAxis
from C_FeatureExtraction.medial_axis_len import Curve
from C_FeatureExtraction.feature_extractions_constants import *
import C_FeatureExtraction.utils as feature_extractor_utils
from a_Common.my_logger import get_new_logger


class BandageFeature:
    def __init__(self, ch_image, ch_outs_dir_name=None):
        self.image_path = ch_image
        self.logger = get_new_logger(self.__class__.__name__)
        if ch_outs_dir_name is not None:
            self.__chromosome_outs_root_dir = ch_outs_dir_name
        else:
            self.__chromosome_outs_root_dir = os.path.join(os.path.dirname(self.image_path),
                                                           os.path.basename(self.image_path).split(".")[
                                                               0] + "_bandage_profile_helper_dir")
        self.__medial_axis_computer = None
        self.medial_axis_curve = None
        self.polynomial_function_symbolic = None
        self.symbol = None
        # array like [((x0,y0), poly1d representing perpendicular to medial axis polynomial fct through pct (x0,y0)), ...]
        self.perpendiculars_array = list()
        self.tangents_array = list()
        self.segments_array = list()
        self.bandage_profile = list()
        self.function_x_points = list()
        self.function_y_points = list()
        self.intersection_points = list()
        self.__helper_dir = os.path.join(self.__chromosome_outs_root_dir, CHROMOSOME_LEN_HELPER_DIR)
        self.bandage_out_file = os.path.join(self.__chromosome_outs_root_dir,
                                             os.path.basename(self.image_path).split(".")[0] + "_bandages.txt")
        self.bandage_out_img_path = os.path.join(self.__chromosome_outs_root_dir,
                                                 os.path.basename(self.image_path).split(".")[0] + "_bandages.png")
        # creez outs dir pentru medial axis
        self.__get_or_create_outs_dir()
        # creez medial axis obj
        self.__medial_axis_computer = MedialAxis(self.image_path, self.__helper_dir)
        # iau path spre imaginea alb-negru
        self.__bin_inv_img_path = self.__medial_axis_computer.get_bin_inv_img_path()
        self.orig_threshold_img = cv2.imread(self.__bin_inv_img_path, 0)

    # @feature_extractor_utils.timing
    def get_bandage_profile(self, intersection_points_nr=None, reevaluate=True):
        self.logger.debug("Start extracting banding profile for {}. Containing {} bandages".format(
            self.image_path, intersection_points_nr))
        if intersection_points_nr is None:
            intersection_points_nr = self.orig_threshold_img.shape[0]
        if os.path.exists(self.bandage_out_file) and not reevaluate:
            self.load_bandage_profile_from_file()
            if not (os.path.exists(self.bandage_out_img_path)):
                self.save_bandage_profile_image()
            return self.bandage_profile
        self.polynomial_function_symbolic = self.__get_medial_axis_polynomial_function_symbolic()
        self.perpendiculars_array = self.get_perpendiculars_array(intersection_points_nr)
        self.segments_array = self.get_segments_array(intersection_points_nr)
        self.bandage_profile = [list() for _ in range(len(self.segments_array))]
        orig_img = cv2.imread(self.image_path, 0)
        progres = 0
        for i in range(self.orig_threshold_img.shape[0]):
            for j in range(self.orig_threshold_img.shape[1]):
                if self.orig_threshold_img[i][j] == 255:
                    index = self.__get_closest_segment_index(feature_extractor_utils.Point(i, j))
                    self.bandage_profile[index].append(orig_img[i][j])
                progres += 1
                if round(100 * progres / (
                        self.orig_threshold_img.shape[0] * self.orig_threshold_img.shape[1])) % 10 == 0:
                    print("Progress = " + str(
                        100 * progres / (self.orig_threshold_img.shape[0] * self.orig_threshold_img.shape[1])))
        # elimin intrarile pentru care nu s-au gasit
        self.bandage_profile = [(sum(x) / len(x)) / 255 for x in self.bandage_profile if len(x) > 0]  # else 255
        self.dump_bandage_profile_in_file()
        self.save_bandage_profile_image()
        return self.bandage_profile

    def compute_medial_axis(self):
        self.__get_medial_axis_polynomial_function_symbolic()

    def __get_medial_axis_polynomial_function_symbolic(self):
        if self.polynomial_function_symbolic is not None:
            return self.polynomial_function_symbolic
        # determin medial axis si iau path spre img
        # curve_path = self.__medial_axis_computer.get_bin_inv_img_path()
        curve_path = self.__medial_axis_computer.get_medial_axis_img()
        if not Curve.is_valid_curve(curve_path):
            curve_path = self.__medial_axis_computer.get_bin_inv_img_path()
        # creez curve obj
        self.medial_axis_curve = Curve(curve_path, grade=CURVE_POLYNOMIAL_FUNCTION_GRADE)
        # iau functia polinomiala pentru medial axis si symbolul ei
        self.polynomial_function_symbolic, self.symbol = self.medial_axis_curve.compute_polynomial_function_symbolic()
        # determin o serie de puncte pentru a reprezenta grafic functia si pentru a determina derivate in ele

        potential_x_points = [round(i) for i in linspace(0, self.orig_threshold_img.shape[0] - 1,
                                                         self.orig_threshold_img.shape[0])]
        for potential_x_point in potential_x_points:
            potential_y_point = int(round(self.medial_axis_curve.polynomial_function(potential_x_point)))
            if potential_y_point < self.orig_threshold_img.shape[1]:
                if self.orig_threshold_img[potential_x_point][potential_y_point] == 255:
                    self.function_x_points.append(potential_x_point)
                    self.function_y_points.append(potential_y_point)

        return self.polynomial_function_symbolic

    def __get_or_create_outs_dir(self):
        if not os.path.exists(self.__helper_dir):
            os.makedirs(self.__helper_dir)

    def get_perpendiculars_array(self, intersection_points_nr):
        self.perpendiculars_array = list()
        self.tangents_array = list()
        deriv_function = np.polyder(self.medial_axis_curve.polynomial_function, 1)
        intersection_points_indexes = linspace(0, len(self.function_x_points) - 1, intersection_points_nr)
        intersection_points_list = list(zip(self.function_x_points, self.function_y_points))
        for i in intersection_points_indexes:
            x0, y0 = intersection_points_list[round(i)]
            # construiesc ecuatia perpendicularei, stiind panta tangentei in pct x0, y0 si punctul(x0,y0)
            # ecuatia dreptei determinata de un pct si o dreapta: y-y0=m(x-x0) => y = m*x -m*x0 +y0
            # pentru 2 drepte perpendiculare, avem: m1*m2 = -1 => panta perpendicularei = -1/(derivata in x0)
            tg_m = deriv_function(x0)
            perpendicular = np.poly1d([tg_m, -tg_m * x0 + y0])
            self.tangents_array.append(((x0, y0), perpendicular))

            m = -1 / tg_m
            perpendicular = np.poly1d([m, -m * x0 + y0])
            self.perpendiculars_array.append(((x0, y0), perpendicular))

        return self.perpendiculars_array

    def get_segments_array(self, intersection_points_nr):
        self.segments_array = list()
        if len(self.perpendiculars_array) == 0:
            self.get_perpendiculars_array(intersection_points_nr)
        for entry in self.perpendiculars_array:
            pct = entry[0]
            max_x = self.orig_threshold_img.shape[0] - 1
            max_y = self.orig_threshold_img.shape[1] - 1
            # if not (0 <= int(pct[0]) <= max_x and 0 <= int(pct[1]) <= max_y):
            #     continue
            line = entry[1]
            m = line.coefficients[0]
            intersection_point = feature_extractor_utils.Point(pct[0], pct[1])
            segment_lower_end_point = feature_extractor_utils.Point(pct[0], pct[1])
            segment_upper_end_point = feature_extractor_utils.Point(pct[0], pct[1])
            found_ch_pixel = False
            if self.orig_threshold_img[int(pct[0])][int(pct[1])] == 255:
                found_ch_pixel = True

            if math.pi / 4 <= m <= 3 * math.pi / 4:
                # unghiul format cu Ox < 45
                # variez x
                # caut capatul segmentului prin scaderea x-ului
                while segment_lower_end_point.x >= DETERMINE_SEGMENT_STEP:
                    segment_lower_end_point.x -= DETERMINE_SEGMENT_STEP
                    segment_lower_end_point.y = line(segment_lower_end_point.x)
                    if 0 <= int(segment_lower_end_point.y) <= max_y:
                        is_white_pixel = self.orig_threshold_img[int(segment_lower_end_point.x)][int(
                            segment_lower_end_point.y)] == 255
                        if is_white_pixel or not found_ch_pixel:
                            if is_white_pixel:
                                found_ch_pixel = True
                            continue
                    # revin la ultimul punct valid
                    segment_lower_end_point.x += DETERMINE_SEGMENT_STEP
                    segment_lower_end_point.y = line(segment_lower_end_point.x)
                    break
                # caut capatul segmentului prin cresterea x-ului
                while int(segment_upper_end_point.x + DETERMINE_SEGMENT_STEP) <= max_x:
                    segment_upper_end_point.x += DETERMINE_SEGMENT_STEP
                    segment_upper_end_point.y = line(segment_upper_end_point.x)
                    if 0 <= int(segment_upper_end_point.y) <= max_y:
                        if self.orig_threshold_img[int(segment_upper_end_point.x)][int(segment_upper_end_point.y)] \
                                == 255:
                            continue
                    # revin la ultimul punct valid
                    segment_upper_end_point.x -= DETERMINE_SEGMENT_STEP
                    segment_upper_end_point.y = line(segment_upper_end_point.x)
                    break
                if segment_lower_end_point != segment_upper_end_point:
                    self.segments_array.append(feature_extractor_utils.Segment(segment_lower_end_point,
                                                                               segment_upper_end_point, line,
                                                                               intersection_point))
            else:
                # unghiul format cu Ox < 45
                # variez y
                # caut capatul segmentului prin scaderea y-ului
                while segment_lower_end_point.y >= DETERMINE_SEGMENT_STEP:
                    segment_lower_end_point.y -= DETERMINE_SEGMENT_STEP
                    # x = -b/a + y/a
                    segment_lower_end_point.x = -line.coefficients[1] / line.coefficients[0] \
                                                + segment_lower_end_point.y / line.coefficients[0]
                    if 0 <= int(segment_lower_end_point.x) <= max_x:
                        if self.orig_threshold_img[int(segment_lower_end_point.x)][int(segment_lower_end_point.y)] \
                                == 255:
                            continue
                    # revin la ultimul punct valid
                    segment_lower_end_point.y += DETERMINE_SEGMENT_STEP
                    # x = -b/a + y/a
                    segment_lower_end_point.x = -line.coefficients[1] / line.coefficients[0] \
                                                + segment_lower_end_point.y / line.coefficients[0]
                    break
                # caut capatul segmentului prin cresterea y-ului
                while int(segment_upper_end_point.y + DETERMINE_SEGMENT_STEP) <= max_y:
                    segment_upper_end_point.y += DETERMINE_SEGMENT_STEP
                    # x = -b/a + y/a
                    segment_upper_end_point.x = -line.coefficients[1] / line.coefficients[0] \
                                                + segment_upper_end_point.y / line.coefficients[0]
                    if 0 <= int(segment_upper_end_point.x) <= max_x:
                        if self.orig_threshold_img[int(segment_upper_end_point.x)][int(segment_upper_end_point.y)] \
                                == 255 and found_ch_pixel:
                            continue
                    # revin la ultimul punct valid
                    segment_upper_end_point.y -= DETERMINE_SEGMENT_STEP
                    # x = -b/a + y/a
                    segment_upper_end_point.x = -line.coefficients[1] / line.coefficients[0] \
                                                + segment_upper_end_point.y / line.coefficients[0]
                    break
                self.segments_array.append(feature_extractor_utils.Segment(segment_lower_end_point,
                                                                           segment_upper_end_point,
                                                                           line,
                                                                           intersection_point))
                if segment_lower_end_point == segment_upper_end_point:
                    print("-------------------")
                    print(str(entry[0]))
                    print(str(entry[1]))
                    print("-------------------")
        if len(self.segments_array) != len(self.perpendiculars_array):
            raise ValueError("len(self.segments_array) != len(self.perpendiculars_array); %d != %d" %
                             (len(self.segments_array), len(self.perpendiculars_array)))
        return self.segments_array

    def __get_closest_segment_index(self, point):
        # todo  poate fi imbunatatita daca fac cautare binara (verific de ce parte a segmentului se afla punctul)
        closest_segment_index = -1
        min_dist = math.inf
        start_search_index = 0
        min_dist_to_intersection = abs(point.y - self.segments_array[start_search_index].intersection_point.y)
        for i in range(1, len(self.segments_array)):
            if min_dist_to_intersection > abs(point.y - self.segments_array[i].intersection_point.y):
                min_dist_to_intersection = abs(point.y - self.segments_array[i].intersection_point.y)
                start_search_index = i
        i_max_limit = max(start_search_index, len(self.segments_array) - start_search_index + 1)
        a = 0
        for i in range(i_max_limit):
            if start_search_index - i >= 0:
                curr_dist = self.segments_array[start_search_index - i].shortest_dist_to_point(point)
                if curr_dist < min_dist:
                    a = 0
                    closest_segment_index = start_search_index - i
                    min_dist = curr_dist
            if start_search_index + i < len(self.segments_array):
                curr_dist = self.segments_array[start_search_index + i].shortest_dist_to_point(point)
                if curr_dist < min_dist:
                    a = 0
                    closest_segment_index = start_search_index + i
                    min_dist = curr_dist
            if a > 5:
                break
            a += 1
        return closest_segment_index

    def __get_closest_segment_index_v1(self, point):
        # todo  poate fi imbunatatita daca fac cautare binara (verific de ce parte a segmentului se afla punctul)
        closest_segment_index = -1
        min_dist = math.inf
        for i in range(len(self.segments_array)):
            curr_dist = self.segments_array[i].shortest_dist_to_point(point)
            if curr_dist < min_dist:
                closest_segment_index = i
                min_dist = curr_dist
        return closest_segment_index

    def dump_bandage_profile_in_file(self):
        with open(self.bandage_out_file, 'wb') as f:
            pickle.dump(self.bandage_profile, f)

        with open(self.bandage_out_file + "_not_serialized.txt", 'w') as f:
            f.write("\n".join([str(x) for x in self.bandage_profile]))

    def save_bandage_profile_image(self):
        # from PIL import Image
        # im = Image.fromarray(np.array([self.bandage_profile for _ in range(30)]))
        img = cv2.imread(self.image_path, 0)
        x_multiplier = round(img.shape[1] / len(self.bandage_profile))
        y_multiplier = img.shape[0]
        one_line = [[x * 255, x * 255, x * 255] for x in self.bandage_profile]
        cv2.imwrite(self.bandage_out_img_path,
                    np.array([[[x * 255, x * 255, x * 255] for x in self.bandage_profile]
                              for _ in range(cv2.imread(self.image_path, 0).shape[1])]))
        # im.show()
        # im.save(self.bandage_out_img_path)

    def load_bandage_profile_from_file(self):
        with open(self.bandage_out_file, 'rb') as f:
            self.bandage_profile = pickle.load(f)

    def get_intersection_points(self):
        """

        :return: list containing points from polyfit function, that will represent
        intersection points between polyfit and perpendiculars / segments
         (Will be considered only those points that are located inside chromosome)
        """
        self.intersection_points = list()
        # nr of rows because img for perpendiculars will be rotated
        if self.medial_axis_curve is None:
            self.__get_medial_axis_polynomial_function_symbolic()
        for x in range(self.orig_threshold_img.shape[0]):
            y = int(round(self.medial_axis_curve.polynomial_function(x)))
            if y < self.orig_threshold_img.shape[1]:
                if self.orig_threshold_img[x][y] == 255:
                    self.intersection_points.append(feature_extractor_utils.Point(x, y))

        return self.intersection_points


def test1(obj):
    import matplotlib.pyplot as plt

    a = 0
    img_bg = cv2.imread(obj.image_path)
    img_bg = imutils.rotate_bound(img_bg, 90)[:, 1:, :]
    # img_bg = imutils.rotate_bound(obj.orig_threshold_img, 90)
    plt.imshow(img_bg, zorder=0)
    for i, j in zip(obj.perpendiculars_array, obj.tangents_array):
        if a % 1 == 0:
            x = obj.function_x_points
            y = i[1](x)
            # plt.plot(x, y, zorder=1)
            # print(str(i[1].coefficients[0]))
            print(str(i[1]))
            print(feature_extractor_utils.get_dist_from_point_to_segment((0, 0), i[1]))
            # x = obj.function_x_points
            # y = j[1](x)
            # plt.plot(x, y)
            # print(str(j[1].coefficients[1]))
            print(str(j[1]))
        a += 1
    # sys.exit()
    # x = obj.medial_axis_curve.polynomial_function
    # xx = x.deriv()
    # print(np.polyder(x, 1)(0))
    # t = 1
    # t = obj.symbol
    # print(x.evalf())
    # print(t)
    # obj.function_x_points = linspace(0, obj.medial_axis_curve.orig_threshold_img.shape[0],
    #                                  obj.medial_axis_curve.orig_threshold_img.shape[0])
    # y = [x.evalf(subs={t: i}) for i in tt]
    # plt.figure(figsize=(10, 10))
    plt.ylim((-200, 250))
    plt.xlim((-10, 170))
    plt.plot(obj.medial_axis_curve.x_points, obj.medial_axis_curve.y_points, 'o', zorder=1)
    plt.plot(obj.medial_axis_curve.polynomial_function(obj.function_x_points), 'r-', zorder=1, linewidth=3)

    # plt.plot(tt, y, '-')
    plt.show()


def test2(obj):
    import matplotlib.pyplot as plt
    f, axarr = plt.subplots(2, 1)
    # axarr[0, 1].imshow(image_datas[1])

    a = 0
    img_bg = cv2.imread(obj.image_path)
    img_bg = imutils.rotate_bound(img_bg, 90)[:, 1:, :]
    img_bg = cv2.flip(img_bg, 1)
    axarr[0].imshow(img_bg, zorder=0)
    for entry in obj.segments_array:
        print("up_x=" + str(entry.a.x) + ";up_y=" + str(entry.a.y) + " - lo_x=" + str(entry.b.x) + ";lo_y=" + str(
            entry.b.y) + " - line=" + str(entry.line))
        x = linspace(max([entry.a.x, entry.b.x]), min(entry.a.x, entry.b.x), 20)
        y = entry.line(x)
        axarr[0].plot(x, y, zorder=3)
        # plt.plot([tuple(entry[0]), tuple(entry[1])], zorder=2)
        # break
    if obj.medial_axis_curve is None:
        obj.compute_medial_axis()
    # plt.axis('off')
    axarr[0].plot(obj.medial_axis_curve.polynomial_function(obj.function_x_points), '-', zorder=1, )
    # pls.savefig(obj.image_path + "_segments.png")
    plt.show()
    col = []
    for i in obj.bandage_profile:
        for j in range(4):
            col.append([int(255*i) for _ in range(3)])

    bands = np.array([col for _ in range(30)])
    axarr[1].imshow(bands)
    plt.show()


def test3(obj):
    for i in range(len(obj.bandage_profile)):
        print(str(i) + " - " + str(obj.bandage_profile[i]))


if __name__ == "__main__":
    obj = BandageFeature(r'__disertation_experiments\dataset\1\1\contrast_split\straight\13-0.bmp')
    band_prof = obj.get_bandage_profile(30)
    print(band_prof)
    # test1(obj)
    test2(obj)
    # test3(obj)
