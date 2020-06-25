import os
from typing import List

import numpy as np

import A_Segmentation.common_operations as common_operations
from A_Segmentation.color_image import ChromosomeFrame
from A_Segmentation.constants import *
from a_Common.my_logger import LOGGER


class ImageSpliter:

    def __init__(self, orig_img_path, colored_img_path, segments_list, inverted=False):
        self.logger = LOGGER.getChild("segmentation")
        self.orig_image_path = orig_img_path
        self.colored_image_path = colored_img_path
        self.segments_list: List[ChromosomeFrame] = segments_list
        self.orig_image = common_operations.read_image(self.orig_image_path, inverted)
        self.colored_image = common_operations.read_image(self.colored_image_path, inverted)
        self.output_dir = os.path.join(os.path.dirname(self.orig_image_path),
                                       ".".join(os.path.basename(self.orig_image_path).split(".")[:-1]) + "_split")
        self.individual_dir = os.path.join(self.output_dir, "individual")
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        if not os.path.exists(self.individual_dir):
            os.makedirs(self.individual_dir)

    def split_image(self):
        segment_index = 0
        for segment in self.segments_list:
            segment_orig = self.orig_image[segment.low_x:segment.up_x, segment.low_y:segment.up_y].copy()

            common_operations.write_image(os.path.join(self.output_dir, str(segment_index) + "_w.bmp"), segment_orig)
            segment_colored = self.colored_image[segment.low_x:segment.up_x, segment.low_y:segment.up_y]
            x_l, y_l = segment_orig.shape[0], segment_orig.shape[1]
            for i in range(x_l):
                for j in range(y_l):
                    if not common_operations.almost_eq_pixels(segment_colored[i][j], segment.color):
                        segment_orig[i][j] = WHITE_COLOR_CODE

            common_operations.write_image(os.path.join(self.output_dir, str(segment_index) + ".bmp"), segment_orig)
            common_operations.write_image(os.path.join(self.individual_dir, str(segment_index) + ".bmp"), segment_orig)
            common_operations.write_image(os.path.join(self.output_dir, str(segment_index) + "_c.bmp"), segment_colored)
            segment_index += 1
