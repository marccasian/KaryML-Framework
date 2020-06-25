import numpy as np
import cv2
import os
from matplotlib import pyplot as plt


class ImagePreProcessor:
    def __init__(self, img_path=r'D:\GIT\Licenta\1stStep\NoiseRemoval\imgs\8067_01-02_251111093037.JPG'):
        self.img_path = img_path
        self.img = cv2.imread(img_path)
        self.dst_dir = ".".join(self.img_path.split(".")[:-1])
        if not os.path.exists(self.dst_dir):
            os.makedirs(self.dst_dir)

    def process(self):
        clahe = cv2.createCLAHE(clipLimit=3., tileGridSize=(8, 8))
        # save orig
        cv2.imwrite(os.path.join(self.dst_dir, 'orig.BMP'), self.img)

        lab = cv2.cvtColor(self.img, cv2.COLOR_BGR2LAB)  # convert from BGR to LAB color space
        l, a, b = cv2.split(lab)  # split on 3 different channels

        l2 = clahe.apply(l)  # apply CLAHE to the L-channel

        lab = cv2.merge((l2, a, b))  # merge channels
        img2 = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)  # convert from LAB to BGR
        cv2.imwrite(os.path.join(self.dst_dir, 'contrast.BMP'), img2)

        # noise removal
        noise_removal = cv2.fastNlMeansDenoisingColored(self.img, None, 10, 10, 7, 21)

        cv2.imwrite(os.path.join(self.dst_dir, 'orig_noise_removal.BMP'), noise_removal)
        # import sys
        # sys.exit(1)
        # erosion

        kernel = np.ones((2, 2), np.uint8)
        erosion = cv2.erode(noise_removal, kernel, iterations=1)
        cv2.imwrite(os.path.join(self.dst_dir, 'erosion.BMP'), erosion)

        # dilation
        dilation = cv2.dilate(noise_removal, kernel, iterations=1)
        cv2.imwrite(os.path.join(self.dst_dir, 'dilatation.BMP'), dilation)

        # erosion2
        kernel = np.ones((10, 10), np.uint8)
        erosion = cv2.erode(noise_removal, kernel, iterations=1)
        cv2.imwrite(os.path.join(self.dst_dir, 'erosion2.BMP'), erosion)

        # dilation2
        dilation1 = cv2.dilate(noise_removal, kernel, iterations=1)
        cv2.imwrite(os.path.join(self.dst_dir, 'dilatation2.BMP'), dilation1)

        # noise removal 2
        noise_removal2 = cv2.fastNlMeansDenoisingColored(dilation, None, 10, 10, 7, 21)

        cv2.imwrite(os.path.join(self.dst_dir, 'noise_removal_after_dilation.BMP'), noise_removal2)

        # segmentation

        gray = cv2.cvtColor(noise_removal2, cv2.COLOR_BGR2GRAY)

        # binary thresholding
        for i in range(0, 255, 5):
            ret, thresh = cv2.threshold(gray, i, 255, cv2.THRESH_BINARY_INV)
            if i == 245:
                cv2.imwrite(os.path.join(self.dst_dir, "segmentation.BMP"), thresh)
            else:
                cv2.imwrite(os.path.join(self.dst_dir, "segmentation_%d.BMP" % i), thresh)

        # Otsu threasholding
        # ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY_INV)
        # cv2.imwrite(os.path.join(self.dst_dir, "segmentation.BMP"), thresh)

        # erosion

        kernel = np.ones((5, 5), np.uint8)
        erosion = cv2.erode(thresh, kernel, iterations=4)

        cv2.imwrite(os.path.join(self.dst_dir, 'erosion_after_segmentation.BMP'), thresh)

        #
        # # 2
        # # noise removal
        # kernel = np.ones((3, 3), np.uint8)
        # opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
        # cv2.imwrite(r'D:\GIT\Licenta\1stStep\NoiseRemoval\imgs\2\2_2.JPG', opening)
        # # sure background area
        # sure_bg = cv2.dilate(opening, kernel, iterations=3)
        # cv2.imwrite(r'D:\GIT\Licenta\1stStep\NoiseRemoval\imgs\2\2_3.JPG', sure_bg)
        # # Finding sure foreground area
        # dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
        # cv2.imwrite(r'D:\GIT\Licenta\1stStep\NoiseRemoval\imgs\2\2_4.JPG', dist_transform)
        # ret, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
        # cv2.imwrite(r'D:\GIT\Licenta\1stStep\NoiseRemoval\imgs\2\2_5_1.JPG', ret)
        # cv2.imwrite(r'D:\GIT\Licenta\1stStep\NoiseRemoval\imgs\2\2_5.JPG', sure_fg)
        # # Finding unknown region
        # sure_fg = np.uint8(sure_fg)
        # unknown = cv2.subtract(sure_bg, sure_fg)
        # cv2.imwrite(r'D:\GIT\Licenta\1stStep\NoiseRemoval\imgs\2\2_6.JPG', unknown)
        #
        #
        # # 3
        # # Marker labelling
        # ret, markers = cv2.connectedComponents(sure_fg)
        # cv2.imwrite(r'D:\GIT\Licenta\1stStep\NoiseRemoval\imgs\2\2_7.JPG', markers)
        # cv2.imwrite(r'D:\GIT\Licenta\1stStep\NoiseRemoval\imgs\2\2_7_1.JPG', ret)
        # # Add one to all labels so that sure background is not 0, but 1
        # markers = markers + 1
        # # Now, mark the region of unknown with zero
        # markers[unknown == 255] = 0
        #
        #
        # # 4
        # markers = cv2.watershed(self.img, markers)
        # self.img[markers == -1] = [255, 0, 0]
        # cv2.imwrite(r'D:\GIT\Licenta\1stStep\NoiseRemoval\imgs\2\2_7.JPG', markers)
