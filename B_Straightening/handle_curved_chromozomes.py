import math
import shutil
from cv2 import cv2
import os
import numpy as np

import A_Segmentation.constants as constants
import A_Segmentation.common_operations as common_operations
import B_Straightening.rotate_image as rotate_image
import B_Straightening.compute_projection_vector as compute_projection_vector


class HandleCurvedChromosome:
    def __init__(self):
        self.image_path = None
        self.current_angle = 0
        self.best_angle = 0
        self.best_score = math.inf
        self.best_image_path = None
        self.best_horizontal_projection_vector = None
        self.p1 = None  # highest point in horizontal projection vector
        self.p2 = None  # second highest point in horizontal projection vector
        self.p3 = None  # min value between p1 and p2 in horizontal projection vector
        self.w1 = 0.5  # tuning parameters to control the weight of each term in the rotation score S
        self.w2 = 0.5  # where w1 < 1, w2 < 1 and w1 + w2 = 1
        self.best_min_position = None
        self.check_if_curved = IsCurvedChromosome()
        self.straighten_curved_chromosome = StraightenCurvedChromosome()
        self.ready_dir = None

    def handle_one_image(self, image_path, straighten=True):
        self.ready_dir = self.get_or_create_ready_dir(image_path)
        if self.check_if_curved.is_curved(image_path):
            for self.current_angle in range(0, 180, 5):
                rotate_image_obj = rotate_image.RotateImage()
                rotate_image_obj.rotate_image(image_path, self.current_angle)
                rotated_image_path = rotate_image_obj.get_output_image_path()
                compute_projection_vector_obj = compute_projection_vector.ComputeProjectionVector()
                h_vector = compute_projection_vector_obj.get_horizontal_projection_vector(rotated_image_path)
                score, min_position = self.__compute_horizontal_vector_score(h_vector)
                if score < self.best_score:
                    self.best_score = score
                    self.best_angle = self.current_angle
                    self.best_min_position = min_position
                    self.best_image_path = rotated_image_path
            if straighten:
                self.straighten_curved_chromosome.straight_curved_chromosome(self.best_image_path,
                                                                             self.best_min_position,
                                                                             self.ready_dir)
            else:
                shutil.copy2(self.best_image_path, self.ready_dir)
            return True
        else:
            shutil.copy2(image_path, self.ready_dir)
            return False

    @staticmethod
    def __compute_score(p1, p2, p3, w1, w2):
        r1 = abs(p1 - p2) / (p1 + p2)
        r2 = p3 / (p1 + p2)
        return w1 * r1 + w2 * r2

    def __compute_horizontal_vector_score(self, h_vector):
        score = math.inf
        min_position = None
        max_position = h_vector.index(max(h_vector))
        max_value = max(h_vector)
        for i in range(len(h_vector)):
            if i != max_position:
                start = min(i, max_position)
                end = max(i, max_position)
                if end - start > 1:
                    min_value = min(h_vector[start + 1:end])
                    current_min_position = h_vector[start + 1:end].index(min_value)
                    current_score = self.__compute_score(max_value, h_vector[i], min_value, self.w1, self.w2)
                    if current_score < score:
                        score = current_score
                        min_position = current_min_position + start + 1
                else:
                    continue
        return score, min_position

    @staticmethod
    def __trim_list(h_vector):
        h_vector_trim = h_vector
        while h_vector_trim[len(h_vector_trim) - 1] == 0:
            h_vector_trim = h_vector_trim[:-1]

        while h_vector_trim[0] == 0:
            h_vector_trim = h_vector_trim[1:]
        return h_vector_trim

    @staticmethod
    def get_or_create_ready_dir(image_path):
        ready_dir_path = os.path.join(os.path.dirname(os.path.dirname(image_path)), "ready")
        if not os.path.exists(ready_dir_path):
            os.makedirs(ready_dir_path)
        return ready_dir_path


class IsCurvedChromosome:
    def __init__(self):
        self.image_path = None

    def is_curved(self, image_to_check_path):
        self.image_path = image_to_check_path
        image = common_operations.read_image(self.image_path)
        compute_projection_vector_obj = compute_projection_vector.ComputeProjectionVector()
        h_vector = compute_projection_vector_obj.get_horizontal_projection_vector(self.image_path)
        ch_area = sum(h_vector)
        total_area = image.shape[0] * image.shape[1]
        proc = ch_area / total_area
        if proc < 0.6:
            return True
        return False


class StraightenCurvedChromosome:
    def __init__(self):
        self.ready_dir_path = None
        self.image_path = None
        self.img = None
        self.img1 = None
        self.img2 = None
        self.img1_path = None
        self.img2_path = None
        self.best_img1_path = None
        self.best_img2_path = None
        self.best_angle_1 = None
        self.best_angle_2 = None

    def straight_curved_chromosome(self, image_path, cut_point, ready_dir_path):
        self.image_path = image_path
        self.ready_dir_path = ready_dir_path
        self.img = common_operations.read_image(self.image_path)
        self.img1 = self.img[:cut_point, :]
        self.img2 = self.img[cut_point:, :]
        img_dir_path = os.path.join(
            os.path.dirname(self.image_path),
            os.path.splitext(os.path.basename(self.image_path))[0])
        img1_dir_path = os.path.join(img_dir_path, "img1")
        img2_dir_path = os.path.join(img_dir_path, 'img2')
        if not os.path.exists(img_dir_path):
            os.makedirs(img_dir_path)
        if not os.path.exists(img1_dir_path):
            os.makedirs(img1_dir_path)
        if not os.path.exists(img2_dir_path):
            os.makedirs(img2_dir_path)
        self.img1_path = os.path.join(
            img1_dir_path,
            os.path.splitext(os.path.basename(self.image_path))[0]
            + "_1"
            + os.path.splitext(os.path.basename(self.image_path))[1])
        self.img2_path = os.path.join(
            img2_dir_path,
            os.path.splitext(os.path.basename(self.image_path))[0]
            + "_2"
            + os.path.splitext(os.path.basename(self.image_path))[1])

        cv2.imwrite(self.img1_path, self.img1)
        cv2.imwrite(self.img2_path, self.img2)

        self.best_img1_path, self.best_angle_1 = self.straight_half_chromosome(self.img1_path)
        self.best_img2_path, self.best_angle_2 = self.straight_half_chromosome(self.img2_path)
        self.unify_chromosomes_parts(os.path.join(self.ready_dir_path, os.path.basename(self.image_path)))

    def straight_half_chromosome(self, img_path):
        rotate_image_object = rotate_image.RotateImage()
        compute_projection_vector_object = compute_projection_vector.ComputeProjectionVector()
        best_angle = 0
        rotate_image_object.rotate_image(img_path, best_angle)
        best_image_path = rotate_image_object.get_output_image_path()
        best_horizontal_projection_vector = compute_projection_vector_object.get_horizontal_projection_vector(img_path)
        best_vertical_projection_vector = compute_projection_vector_object.get_vertical_projection_vector(img_path)
        best_score = self.get_orientation_score(best_horizontal_projection_vector, best_vertical_projection_vector)
        for current_angle in range(0, 180, 5):
            rotate_image_object.rotate_image(img_path, current_angle)
            rotated_image_path = rotate_image_object.get_output_image_path()
            horizontal_projection_vector = \
                compute_projection_vector_object.get_horizontal_projection_vector(rotated_image_path)
            vertical_projection_vector = \
                compute_projection_vector_object.get_vertical_projection_vector(rotated_image_path)
            score = self.get_orientation_score(horizontal_projection_vector, vertical_projection_vector)
            if score < best_score:
                best_score = score
                best_image_path = rotated_image_path
                best_angle = current_angle
        print("Best angle = %d" % best_angle)
        if best_score < math.inf:
            print("Best score = %d" % best_score)
        print("Best image path = %s" % best_image_path)
        return best_image_path, best_angle

    @staticmethod
    def get_orientation_score(horizontal_projection_vector, vertical_projection_vector):
        v = 0
        h = 0
        for i in horizontal_projection_vector:
            if i > 0:
                h += 1
        for i in vertical_projection_vector:
            if i > 0:
                v += 1
        if h > v:
            return h * v
        return math.inf

    def unify_chromosomes_parts(self, out_path):
        img1 = common_operations.read_image(self.best_img1_path)
        img2 = common_operations.read_image(self.best_img2_path)
        if self.best_angle_2 > 90:
            img2 = np.flipud(img2)
            cut2 = 180 - self.best_angle_2
        else:
            cut2 = self.best_angle_2
        cut2 //= 8
        if self.img2.shape[0] > img2.shape[0] - cut2:
            cut2 = max(0, self.img2.shape[0] - img2.shape[0])

        if self.best_angle_1 > 90:
            img1 = np.flipud(img1)
            cut1 = 180 - self.best_angle_1
        else:
            cut1 = self.best_angle_1
        cut1 //= 8
        if self.img1.shape[0] > img1.shape[0] - cut1:
            cut1 = max(0, self.img1.shape[0] - img1.shape[0])
        print(img1.shape)
        print(img2.shape)
        h = int(img1.shape[0]) + int(img2.shape[0])
        if h > cut1 + cut2:
            h = h - cut1 - cut2
        offset1, offset2 = self.__get_imgs_offset(img1, img2, cut1, cut2)
        print("         Img1   Img2")
        print("Shape: %s   %s" % (str(img1.shape), str(img2.shape)))
        print("Cut: %d   %d" % (cut1, cut2))
        print("Offset: %d   %d" % (offset1, offset2))
        w = max(img1.shape[1] + offset1, img2.shape[1] + offset2)
        print("h=%d" % h)
        print("w=%d" % w)
        img3 = np.zeros((h, w, 3))
        img3[:, :] = (255, 255, 255)
        start_row_offset_for_2nd_img = img1.shape[0] - cut1
        if cut1 > 0:
            img3[:img1.shape[0] - cut1, offset1:img1.shape[1] + offset1] = img1[:-cut1, :]
        else:
            img3[:img1.shape[0], offset1:img1.shape[1] + offset1] = img1[:, :]
            start_row_offset_for_2nd_img = img1.shape[0]
        if cut2 > 0:
            img3[start_row_offset_for_2nd_img:, offset2:img2.shape[1] + offset2] = img2[cut2:, :]
        else:
            img3[start_row_offset_for_2nd_img:, offset2:img2.shape[1] + offset2] = img2[:, :]
        cv2.imwrite(out_path, img3)
        print("========================================================")

    def __get_imgs_offset(self, img1, img2, cut1, cut2):
        img1_last_row = img1[img1.shape[0] - max(cut1, 0) - 1]
        img2_first_row = img2[max(0, cut2)]
        img1_left, img1_right = self.__find_limits(img1_last_row)
        img2_left, img2_right = self.__find_limits(img2_first_row)
        img1_mid_offset = ((img1_right - img1_left) // 2) + img1_left
        img2_mid_offset = ((img2_right - img2_left) // 2) + img2_left
        print("Img1 mid offset= %d" % img1_mid_offset)
        print("Img2 mid offset= %d" % img2_mid_offset)
        print("Img1 left %d right %d" % (img1_left, img1_right))
        print("Img2 left %d right %d" % (img2_left, img2_right))
        if img1.shape[1] > img2.shape[1]:
            if img1_mid_offset - img2_mid_offset > 0:
                img1_offset = 0
                img2_offset = img1_mid_offset - img2_mid_offset
            else:
                img1_offset = img2_mid_offset - img1_mid_offset
                img2_offset = 0
        else:
            if img2_mid_offset - img1_mid_offset > 0:
                img1_offset = img2_mid_offset - img1_mid_offset
                img2_offset = 0
            else:
                img1_offset = 0
                img2_offset = img1_mid_offset - img2_mid_offset
        return img1_offset, img2_offset

    @staticmethod
    def __find_limits(list_to_process):
        st = 0
        for st in range(len(list_to_process)):
            check = list_to_process[st] == constants.WHITE_COLOR_CODE
            if check.all():
                continue
            else:
                break
        dr = len(list_to_process)
        for dr in range(len(list_to_process) - 1, st, -1):
            check = list_to_process[dr] == constants.WHITE_COLOR_CODE
            if check.all():
                continue
            else:
                break
        return st, dr


def handle_curved_chromosomes(imgs_dir):
    ready_dir = None
    for path_to_img in common_operations.get_all_images(imgs_dir):
        obj_handle_curved_ch = HandleCurvedChromosome()
        obj_handle_curved_ch.handle_one_image(path_to_img)
        ready_dir = obj_handle_curved_ch.ready_dir
    return ready_dir
