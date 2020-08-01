import os

import A_Segmentation.common_operations as common_operations
from A_Segmentation.constants import *
from a_Common.my_logger import LOGGER


class ChromosomeFrame:
    def __init__(self, up_x, low_x, up_y, low_y, color):
        """
        Describes a frame(image zone coordinates) where a  chromosome is located and the chromosome color.
        :param up_x, low_x, up_y, low_y: x,y frame coordinates
        :param color: Chromosome color code list containing RGB colors
        """
        self.up_x, self.low_x, self.up_y, self.low_y = up_x, low_x, up_y, low_y
        self.color = color


class ColorImage:
    def __init__(self, input_image_path, invert=False):
        self.image_path = input_image_path
        self.image = common_operations.read_image(self.image_path, invert)
        self.segments = list()
        self.colored_image_path = self._get_colored_image_path()
        self.logger = LOGGER.getChild("segmentation")
        self.__color_index = 0
        self.__current_color = [20, 0, 0]
        self.__current_color_index_to_increment = 1

    def update_current_color(self):
        self.__current_color[self.__current_color_index_to_increment] += 20
        self.__current_color_index_to_increment = (self.__current_color_index_to_increment + 1) % 3
        self.__color_index += 1

    def color_image(self):
        self._increase_black_level()
        old_x, old_y = 0, 0
        new_segment_position = self._get_next_pixel_position(MIN_COLOR_CODE_TO_FILL, old_x, old_y)
        while new_segment_position:
            self.logger.debug(new_segment_position)
            max_x, min_x, max_y, min_y = \
                self.fill_segment(new_segment_position[0], new_segment_position[1], MIN_COLOR_CODE_TO_FILL)
            new_segment_position = self._get_next_pixel_position(MIN_COLOR_CODE_TO_FILL, old_x, old_y)
            if new_segment_position is not None:
                old_x, old_y = new_segment_position[0], new_segment_position[1]
            if not self.is_valid_chromosome(max_x, min_x, max_y, min_y):
                self.undo_fill(max_x, min_x, max_y, min_y)
            else:
                self.segments.append(ChromosomeFrame(max_x, min_x, max_y, min_y, COLORS[self.__color_index]))
                self.update_current_color()

        self.save_image()

    def _get_next_pixel_position(self, pixel_code, old_x=0, old_y=0):
        l, c, _ = self.image.shape
        for i in range(old_x, l):
            if i == old_x:
                for j in range(old_y, c):
                    if common_operations.gt_pixels(self.image[i][j], pixel_code):
                        return [i, j]
            else:
                for j in range(c):
                    if common_operations.gt_pixels(self.image[i][j], pixel_code):
                        return [i, j]
        return None

    def _get_next_pixel_lt_position(self, pixel_code, old_x=0, old_y=0):
        l, c, _ = self.image.shape
        for i in range(old_x, l):
            if old_x == i:
                for j in range(old_y, c):
                    if common_operations.lt_pixels(self.image[i][j], pixel_code):
                        return [i, j]
            else:
                for j in range(c):
                    if common_operations.lt_pixels(self.image[i][j], pixel_code):
                        return [i, j]

        return None

    def fill_segment(self, x, y, pixel_code):
        list_to_check = [[x, y]]
        max_x = 0
        min_x = len(self.image)
        max_y = 0
        min_y = len(self.image[0])
        it = 0
        self.logger.debug("shape = %d x %d" % (self.image.shape[0], self.image.shape[1]))
        while len(list_to_check) > 0 and it < self.image.shape[0] * self.image.shape[1]:
            if it % 10000 == 0:
                self.logger.debug("it = %d; len(list_to_check) = %d" % (it, len(list_to_check)))
            it += 1
            x, y = list_to_check.pop()
            if common_operations.gt_pixels(self.image[x][y], pixel_code):
                if max_x < x:
                    max_x = x
                if min_x > x:
                    min_x = x
                if max_y < y:
                    max_y = y
                if min_y > y:
                    min_y = y
                self.image[x][y] = COLORS[self.__color_index]
                if x > 0 and common_operations.gt_pixels(self.image[x - 1][y], pixel_code):
                    if [x - 1, y] not in list_to_check:
                        list_to_check.append([x - 1, y])
                if x < len(self.image) - 1 and common_operations.gt_pixels(self.image[x + 1][y], pixel_code):
                    if [x + 1, y] not in list_to_check:
                        list_to_check.append([x + 1, y])
                if y > 0 and common_operations.gt_pixels(self.image[x][y - 1], pixel_code):
                    if [x, y - 1] not in list_to_check:
                        list_to_check.append([x, y - 1])
                if y < len(self.image[x]) - 1 and common_operations.gt_pixels(self.image[x][y + 1], pixel_code):
                    if [x, y + 1] not in list_to_check:
                        list_to_check.append([x, y + 1])
        return max_x, min_x, max_y, min_y

    def save_image(self):
        common_operations.write_image(self.colored_image_path, self.image)

    def _get_colored_image_path(self):
        return os.path.join(os.path.dirname(self.image_path),
                            ".".join(os.path.basename(self.image_path).split(".")[:-1]) + "_colored."
                            + self.image_path.split(".")[-1])

    def _increase_black_level(self):
        self.logger.debug("increase_black")
        old_pos_x = 0
        old_pos_y = 0
        new_segment_position = self._get_next_pixel_lt_position([10, 10, 10], old_pos_x, old_pos_y)
        self.fill_segment(new_segment_position[0], new_segment_position[1], [0, 0, 0])

    def is_valid_chromosome(self, max_x, min_x, max_y, min_y):
        pixel_crop_area = (max_x - min_x) * (max_y - min_y)
        image_area = self.image.shape[0] * self.image.shape[1]
        if (pixel_crop_area / image_area) * 100 > MINIMUM_PERCENTAGE_OF_CHROMOSOME:
            return True
        return False

    def undo_fill(self, max_x, min_x, max_y, min_y):
        for i in range(min_x, max_x + 1):
            for j in range(min_y, max_y + 1):
                if common_operations.almost_eq_pixels(self.image[i][j], COLORS[self.__color_index]):
                    self.image[i][j] = WHITE_COLOR_CODE
