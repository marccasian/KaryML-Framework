import os
import pickle
import sys
import traceback

from a_Common.my_logger import LOGGER
import A_Segmentation.color_image as color_image
import A_Segmentation.split_image as split_image
import A_Segmentation.pre_process_image as pre_process_image


class ImageSegmenter:
    def __init__(self, image_list=None):
        self.logger = LOGGER.getChild("segmentation")
        self.images = image_list

    def segment_images(self):
        for image in self.images:
            logger.debug("Image: {}".format(image))
            try:
                img_segmenter.process_one_image(image, logger)
            except BaseException as exc:
                self.logger.exception("Exception:{}; Traceback:{}".format(exc, traceback.format_exc()))

    def process_one_image(self,
                          input_img_path=r"D:\GIT\Licenta\1stStep\NoiseRemoval\try_imgs\8094_01-04_011211100401.JPG",
                          invert=False):
        pre_proc_obj = pre_process_image.ImagePreProcessor(input_img_path)
        pre_proc_obj.process()
        dst_dir = pre_proc_obj.dst_dir
        img_path = os.path.join(dst_dir, "segmentation.BMP")
        orig_path = os.path.join(dst_dir, "contrast.BMP")
        segments_file_path = os.path.join(dst_dir, "segments.txt")
        colored_image_path = os.path.join(dst_dir, "segmentation_colored.BMP")

        if not os.path.exists(segments_file_path):
            obj = color_image.ColorImage(img_path, invert)
            self.logger.debug("color image...")
            obj.color_image()
            with open(segments_file_path, "wb") as fp:  # Pickling
                pickle.dump(obj.segments, fp)
            j = 0
            for i in obj.segments:
                self.logger.debug(str(j) + " - " + str(i))
                j += 1

        s_list = list()

        with open(segments_file_path, "rb") as fp:  # Unpickling
            s_list = pickle.load(fp)
        j = 0
        for i in s_list:
            print(str(j) + " - " + str(i))
            j += 1
        obj_split = split_image.ImageSpliter(orig_path, colored_image_path, s_list, invert)
        obj_split.split_image()
        self.logger.debug(dst_dir)
        self.logger.debug(obj_split.individual_dir)
        self.logger.debug(obj_split.output_dir)
        # d:\GIT\Karyotyping-Project\PythonProject\Z_Images\autom\1
        # d:\GIT\Karyotyping-Project\PythonProject\Z_Images\autom\1\contrast_split\individual
        # d:\GIT\Karyotyping-Project\PythonProject\Z_Images\autom\1\contrast_split
        return dst_dir, obj_split.individual_dir, obj_split.output_dir


def get_all_images(dir_path="imgs"):
    return [os.path.join(dir_path, f)
            for f in os.listdir(dir_path)
            if f.lower().endswith('.jpg')
            or f.lower().endswith('.bmp')
            or f.lower().endswith('.jpeg')
            ]


if __name__ == "__main__":
    logger = LOGGER.getChild("segmentation")
    logger.info("cwd={}".format(os.getcwd()))
    images = get_all_images(r'..\Z_Images\test')
    logger.debug("Images: {}".format(images))
    img_segmenter = ImageSegmenter(images)
    img_segmenter.segment_images()