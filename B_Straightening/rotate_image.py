import os

from PIL import Image
import B_Straightening.image_trimmer as image_trimmer


class RotateImage:
    def __init__(self, dump_in_file=True, output_dir=None):
        self.img_path = None
        self.angle = None
        self.dump_in_file = dump_in_file
        self.output_dir = output_dir
        self.output_image_path = None
        self.current_image_dir = None
        self.image_trimmer = image_trimmer.TrimImage()

    def update_class_attributes(self, image, angle):
        if self.output_dir is None:
            self.output_dir = os.path.dirname(image)
        self.current_image_dir = os.path.join(self.output_dir, os.path.basename(image).split('.')[0]) + "-rotations"

        if not os.path.exists(self.current_image_dir):
            os.makedirs(self.current_image_dir)
        self.output_image_path = os.path.join(
            self.current_image_dir,
            os.path.basename(image).split('.')[0]) + "-%d.bmp" % angle
        self.img_path = image
        self.angle = angle

    def rotate_image(self, image, angle):
        self.update_class_attributes(image, angle)
        img = Image.open(self.img_path)
        # converted to have an alpha layer
        im2 = img.convert('RGBA')
        # rotated image
        rot = im2.rotate(self.angle, expand=1)
        # a white image same size as rotated image
        fff = Image.new('RGBA', rot.size, (255,) * 4)
        # create a composite image using the alpha layer of rot as a mask
        out = Image.composite(rot, fff, rot)
        # save your work (converting back to mode='1' or whatever..)
        if self.dump_in_file:
            # print("Dump rotated image in file %s" % self.output_image_path)
            out.convert(img.mode).save(self.output_image_path)
            self.image_trimmer.trim(self.output_image_path)
        return out.convert(img.mode)

    def get_output_image_path(self):
        return self.output_image_path


if __name__ == '__main__':
    img_path = r'D:\GIT\Licenta\1stStep\NoiseRemoval\img_me\8094_01-04_011211100401\contrast_split\3.jpg'
    rotateObj = RotateImage(output_dir='.')
    rotateObj.rotate_image(img_path, 30)
