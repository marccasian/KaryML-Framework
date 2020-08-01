import os

import shutil

import math

import B_Straightening.rotate_image as rotate_image
import B_Straightening.compute_projection_vector as compute_projection_vector


class ChromosomeOrientationUpdater:
    def __init__(self, image_path):
        self.img_path = image_path
        self.current_angle = 0
        self.best_angle = 0
        self.best_score = math.inf
        self.best_image_path = None
        self.best_horizontal_projection_vector = None
        self.best_vertical_projection_vector = None

    def update_chromosome_orientation(self):
        rotate_image_object = rotate_image.RotateImage()
        compute_projection_vector_object = compute_projection_vector.ComputeProjectionVector()
        rotate_image_object.rotate_image(self.img_path, self.current_angle)
        self.best_image_path = rotate_image_object.get_output_image_path()
        self.best_horizontal_projection_vector = \
            compute_projection_vector_object.get_horizontal_projection_vector(self.img_path)
        self.best_vertical_projection_vector = \
            compute_projection_vector_object.get_vertical_projection_vector(self.img_path)
        self.best_score = \
            self.get_orientation_score(self.best_horizontal_projection_vector, self.best_vertical_projection_vector)
        for self.current_angle in range(10, 180, 5):
            rotate_image_object.rotate_image(self.img_path, self.current_angle)
            rotated_image_path = rotate_image_object.get_output_image_path()
            horizontal_projection_vector = \
                compute_projection_vector_object.get_horizontal_projection_vector(rotated_image_path)
            vertical_projection_vector = \
                compute_projection_vector_object.get_vertical_projection_vector(rotated_image_path)
            score = self.get_orientation_score(horizontal_projection_vector, vertical_projection_vector)
            if score < self.best_score:
                self.best_horizontal_projection_vector = horizontal_projection_vector
                self.best_vertical_projection_vector = vertical_projection_vector
                self.best_score = score
                self.best_image_path = rotated_image_path
                self.best_angle = self.current_angle

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


def get_all_images(dir_path="imgs"):
    return [os.path.join(dir_path, f)
            for f in os.listdir(dir_path)
            if f.lower().endswith('.jpg')
            or f.lower().endswith('.bmp')
            or f.lower().endswith('.jpeg')
            ]


def update(imgs_dir_path, selected_imgs_dir_path):
    if not os.path.exists(selected_imgs_dir_path):
        os.makedirs(selected_imgs_dir_path)
    for img_path in get_all_images(imgs_dir_path):
        orientation_updater = ChromosomeOrientationUpdater(img_path)
        orientation_updater.update_chromosome_orientation()
        print(orientation_updater.best_image_path)
        shutil.copy2(orientation_updater.best_image_path, selected_imgs_dir_path)
    return selected_imgs_dir_path
