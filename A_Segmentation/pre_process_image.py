import numpy as np
import cv2
import os


class ImagePreProcessor:
    def __init__(self, img_path):
        self.img_path = img_path
        self.img = cv2.imread(img_path)
        self.dst_dir = ".".join(self.img_path.split(".")[:-1])
        if not os.path.exists(self.dst_dir):
            os.makedirs(self.dst_dir)

    def process(self):
        clahe = cv2.createCLAHE(clipLimit=3., tileGridSize=(8, 8))
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

        # erosion
        kernel = np.ones((5, 5), np.uint8)
        erosion = cv2.erode(thresh, kernel, iterations=4)

        cv2.imwrite(os.path.join(self.dst_dir, 'erosion_after_segmentation.BMP'), thresh)
