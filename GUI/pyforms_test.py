import os
from threading import Thread
import shutil

import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print('[ [{script}] is applying path hack [{path}] before import! ]'.format(script=os.path.abspath(__file__),
                                                                            path=os.path.dirname(os.path.dirname(
                                                                                os.path.abspath(__file__)))))
from confapp import conf
# from pyforms.utils.settings_manager import conf
import GUI.settings as settings

conf += settings
import pprint

import pyforms_gui
from pyforms_gui.basewidget import BaseWidget
from pyforms_gui.basewidget import BaseWidget
from pyforms.controls import ControlFile
from pyforms.controls import ControlButton
from pyforms.controls import ControlLabel
from pyforms.controls import ControlSlider
from pyforms.controls import ControlCheckBox
from a_Common.constants import *


class SimpleExample1(BaseWidget):

    LENGTH_FEATURE = "Chromosome length"
    AREA_FEATURE = "Chromosome area"
    CENTROMERIC_INDEX_FEATURE = "Centromeric Index"
    BANDING_PATTERN_FEATURE = "Banding pattern"
    FEATURES = [
        LENGTH_FEATURE,
        CENTROMERIC_INDEX_FEATURE,
        BANDING_PATTERN_FEATURE
    ]

    def __init__(self):
        super(SimpleExample1, self).__init__('KaryML Framework')
        self._app_title = ControlLabel("KaryML Framework")
        # Definition of the forms fields
        self._input_image_path = ControlFile('Input image')
        self._pairs_path = ControlFile('Expected karyotype (optional)')
        # self._input_image_path.add_popup_menu_option("Image", function_action=self._image_changed)
        # self._input_image = ControlImage()
        # self._test_name = ControlText('Test name')

        self._features_label = ControlLabel("Chose features to be extracted")

        self._f1_check_box = ControlCheckBox(label=self.LENGTH_FEATURE, default=True)
        self._f1_check_box.changed_event = self._f1_check_box_changed
        self._f2_check_box = ControlCheckBox(label=self.CENTROMERIC_INDEX_FEATURE, default=True)
        self._f2_check_box.changed_event = self._f2_check_box_changed
        self._f3_check_box = ControlCheckBox(label=self.BANDING_PATTERN_FEATURE, default=True)
        self._f3_check_box.changed_event = self._f3_check_box_changed
        self._f4_check_box = ControlCheckBox(label=self.AREA_FEATURE, default=True)
        self._f4_check_box.changed_event = self._f4_check_box_changed

        self._eu_dist = ControlButton(EUCLIDEAN_DISTANCE)
        self._we_eu_dist = ControlButton(WEIGHTED_EUCLIDEAN_DISTANCE)
        self._man_dist = ControlButton(MANHATTAN_DISTANCE)

        self._dist_label = ControlLabel(label="Distance to use: " + EUCLIDEAN_DISTANCE.upper(),
                                        default="Distance to use: " + EUCLIDEAN_DISTANCE)

        self._f1_w = ControlSlider(label="Chromosome length", default=25, minimum=0, maximum=100, visible=False)
        self._f2_w = ControlSlider(label="  Centromeric Index", default=25, minimum=0, maximum=100, visible=False)
        self._f3_w = ControlSlider(label="      Banding pattern", default=25, minimum=0, maximum=100, visible=False)
        self._f4_w = ControlSlider(label="Chromosome area", default=25, minimum=0, maximum=100, visible=False)

        self._epochs_no = ControlSlider(label="    Epochs Nr.", default=200000, minimum=50000, maximum=400000)
        self._rows = ControlSlider(label="     Map rows", default=50, minimum=10, maximum=100)
        self._cols = ControlSlider(label="Map columns", default=50, minimum=10, maximum=100)

        self.errors_label_text = ControlLabel(label="Errors:", default="Errors:", visible=True)
        self.errors_label = ControlLabel(label="Errors", default="", visible=False)
        self.info_label_text = ControlLabel(label="Info:", default="Info:", visible=True)
        self.info_label = ControlLabel(label="Info", default="", visible=False)
        self._button = ControlButton('Start KaryML Framework')
        self._button.value = self._runKarySomAction

        self._eu_dist.value = self.__dist_changed_eu
        self._we_eu_dist.value = self.__dist_changed_we_eu
        self._man_dist.value = self.__dist_changed_man
        self.t = None

    def _runKarySomAction(self):
        """Button action event"""
        print("Aici")
        if self.info_label.visible is True:
            self.info_label.hide()
        if self._dist_label.value.split()[len(self._dist_label.value.split()) - 2] == "Weighted":
            dist_lbl = " ".join(self._dist_label.value.split()[len(self._dist_label.value.split()) - 2:])
        else:
            dist_lbl = self._dist_label.value.split()[len(self._dist_label.value.split()) - 1]
        if len([2 ** j for j in range(4) if [self._f1_check_box,
                                             self._f2_check_box,
                                             self._f3_check_box,
                                             self._f4_check_box][j].value]) < 1:
            self.errors_label.value = "At least one feature is required"
            self.errors_label.show()
            return
        else:
            self.errors_label.hide()
        cfg = {
            "distance": dist_lbl,
            "feature_names": [self.FEATURES[i] for i in [j for j in range(3) if
                                                         [self._f1_check_box.value,
                                                          self._f2_check_box.value,
                                                          self._f3_check_box.value,
                                                          self._f4_check_box.value][j]]],
            "features": [2 ** j for j in range(4) if [self._f1_check_box,
                                                      self._f2_check_box,
                                                      self._f3_check_box,
                                                      self._f4_check_box][j].value],
            "epochs": self._epochs_no.value,
            "input": self._input_image_path.value,
            "pairs": self._pairs_path.value,
            "rows": self._rows.value,
            "cols": self._cols.value,
            "weights": None
        }
        if dist_lbl == WEIGHTED_EUCLIDEAN_DISTANCE:

            if self.get_sum_of_weights() != 100:
                print("Error")
                self.errors_label.value = "Sum of weights must be 100"
                self.errors_label.show()
                return

            cfg["weights"] = dict()
            if self._f1_check_box.value:
                cfg["weights"][1] = self._f1_w.value / 100
            if self._f2_check_box.value:
                cfg["weights"][2] = self._f2_w.value / 100
            if self._f3_check_box.value:
                cfg["weights"][3] = self._f3_w.value / 100
            if self._f4_check_box.value:
                cfg["weights"][4] = self._f4_w.value / 100
        if not os.path.exists(self._input_image_path.value):
            self.errors_label.value = "Image path %s doesn't exists" % self._input_image_path.value
            self.errors_label.show()
            return
        self.errors_label.hide()

        pprint.pprint(cfg)
        from run_steps import start_process_for_one_image
        mdist = False
        if dist_lbl == MANHATTAN_DISTANCE:
            mdist = True

        self.t = Thread(target=start_process_for_one_image, args=[cfg["input"], mdist, "", sum(cfg["features"]),
                                                                  cfg["epochs"], cfg["rows"], cfg["cols"],
                                                                  cfg["weights"], self])
        self.t.start()

        msg = ""
        if not os.path.isfile(self._pairs_path.value):
            msg += "Expected karyotype missing or file not exists. PrecKar value won't be calculated."
        else:
            msg += "Karyotype file found. Will compute PrecKar value"
            dir_path = '.'.join(self._input_image_path.value.split('.')[:-1])
            if not os.path.isdir(dir_path):
                os.makedirs(dir_path)
            shutil.copy2(self._pairs_path.value, os.path.join(dir_path, "pairs.txt"))
        self.info_label.value = msg + "\n" + "Start KarySOM for current configuration. Processing..."
        self.info_label.show()
        # self._test_name.value = self._input_image.value

    def __dist_changed_eu(self):
        self._dist_label.value = "Distance to use: " + EUCLIDEAN_DISTANCE
        self._dist_label.label = "Distance to use: " + EUCLIDEAN_DISTANCE.upper()
        self._f1_w.hide()
        self._f2_w.hide()
        self._f3_w.hide()
        self._f4_w.hide()
        self.info_label.hide()

    def __dist_changed_we_eu(self):
        self._dist_label.value = "Distance to use: " + WEIGHTED_EUCLIDEAN_DISTANCE
        self._dist_label.label = "Distance to use: " + WEIGHTED_EUCLIDEAN_DISTANCE.upper()
        if self._f1_check_box.value:
            self._f1_w.show()
        if self._f2_check_box.value:
            self._f2_w.show()
        if self._f3_check_box.value:
            self._f3_w.show()
        if self._f4_check_box.value:
            self._f4_w.show()
        self.info_label.hide()

    def __dist_changed_man(self):
        self._dist_label.value = "Distance to use: " + MANHATTAN_DISTANCE
        self._dist_label.label = "Distance to use: " + MANHATTAN_DISTANCE.upper()
        self._f1_w.hide()
        self._f2_w.hide()
        self._f3_w.hide()
        self._f4_w.hide()
        self.info_label.hide()

    def get_sum_of_weights(self):
        suma = 0
        if self._f1_check_box.value:
            suma += self._f1_w.value
        if self._f2_check_box.value:
            suma += self._f2_w.value
        if self._f3_check_box.value:
            suma += self._f3_w.value
        if self._f4_check_box.value:
            suma += self._f4_w.value
        return suma

    def _f1_check_box_changed(self):
        if self._f1_check_box.value and "weighted" in self._dist_label.value.lower():
            self._f1_w.show()
        else:
            self._f1_w.hide()

    def _f2_check_box_changed(self):
        if self._f2_check_box.value and "weighted" in self._dist_label.value.lower():
            self._f2_w.show()
        else:
            self._f2_w.hide()

    def _f3_check_box_changed(self):
        if self._f3_check_box.value and "weighted" in self._dist_label.value.lower():
            self._f3_w.show()
        else:
            self._f3_w.hide()

    def _f4_check_box_changed(self):
        if self._f4_check_box.value and "weighted" in self._dist_label.value.lower():
            self._f4_w.show()
        else:
            self._f4_w.hide()


# Execute the application
if __name__ == "__main__":
    from pyforms import start_app

    start_app(SimpleExample1)
