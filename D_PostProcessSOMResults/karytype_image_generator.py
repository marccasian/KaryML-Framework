from cv2 import cv2
import numpy as np
import os


class KaryotypeImageGenerator:
    def __init__(self, karyotype, som_results_file):
        """
        karyotype
        [
            [ch1, ch2, {ch3}]
            [ch1, ch2, {ch3}]
            [ch1, ch2, {ch3}]
            ...
        ]
        :param karyotype:
        """
        self.__karyotype = karyotype
        self.__images_was_loaded = False
        self.__max_w = 0
        self.__max_h = 0
        self.__nr_of_lines = len(self.__karyotype)
        self.__nr_of_columns = 0
        self.__karyotype_image = None
        self.karyotype_image_path = som_results_file[:-4] + "_generated_karyotype.bmp"

    def __load_images_object(self):
        if not self.__images_was_loaded:
            for k_entry in self.__karyotype:
                for ch in k_entry:
                    ch.image_obj = cv2.imread(ch.ch_img_path)
            self.__images_was_loaded = True

    def __get_max_ch_dimensions(self):
        self.__load_images_object()
        for k_entry in self.__karyotype:
            if len(k_entry) > self.__nr_of_columns:
                self.__nr_of_columns = len(k_entry)
            for ch in k_entry:
                if not os.path.exists(ch.ch_img_path):
                    return
                h, w, _ = ch.image_obj.shape
                if self.__max_h < h:
                    self.__max_h = h
                if self.__max_w < w:
                    self.__max_w = w

    def generate_karyotype_image(self):
        if self.__max_h == 0 or self.__max_w == 0 or self.__nr_of_columns == 0:
            self.__get_max_ch_dimensions()
        self.__karyotype_image = np.zeros(
            (self.__nr_of_lines * self.__max_h + 1, self.__nr_of_columns * self.__max_w + 1, 3))
        current_i = 0
        for k_entry in self.__karyotype:
            current_j = 0
            for ch in k_entry:
                if not os.path.exists(ch.ch_img_path):
                    return
                print(ch.image_obj.shape)
                h, w, _ = ch.image_obj.shape
                self.__karyotype_image[current_i:current_i + h, current_j:current_j + w] = ch.image_obj
                current_j += self.__max_w
            current_i += self.__max_h
        cv2.imwrite(self.karyotype_image_path, self.__karyotype_image)
