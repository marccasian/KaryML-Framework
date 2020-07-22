import os
from threading import Thread
import shutil

import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print('[ [{script}] is applying path hack [{path}] before import! ]'.format(script=os.path.abspath(__file__),
                                                                            path=os.path.dirname(os.path.dirname(
                                                                                os.path.abspath(__file__)))))
from confapp import conf
import GUI.settings as settings

conf += settings
import pprint

from pyforms_gui.basewidget import BaseWidget
from pyforms.controls import ControlFile
from pyforms.controls import ControlButton
from pyforms.controls import ControlLabel
from pyforms.controls import ControlSlider
from pyforms.controls import ControlCheckBox

from a_Common.constants import *
from run_steps import start_process_for_one_image


class KaryML_Main(BaseWidget):
    LENGTH_FEATURE = "Chromosome length"
    AREA_FEATURE = "Chromosome area"
    CENTROMERIC_INDEX_FEATURE = "Centromeric Index"
    BANDING_PATTERN_FEATURE = "Banding pattern"
    FEATURES = [
        LENGTH_FEATURE,
        CENTROMERIC_INDEX_FEATURE,
        BANDING_PATTERN_FEATURE
    ]
    APP_NAME = 'KaryML Framework'

    def __init__(self):
        super(KaryML_Main, self).__init__(self.APP_NAME)
        self._app_title = ControlLabel(self.APP_NAME)
        self._input_image_path = ControlFile('Input image')
        self._pairs_path = ControlFile('Expected karyotype (optional)')

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
        self._button = ControlButton('Start {}'.format(self.APP_NAME))
        self._button.value = self._runKarySomAction

        self._eu_dist.value = self.__dist_changed_eu
        self._we_eu_dist.value = self.__dist_changed_we_eu
        self._man_dist.value = self.__dist_changed_man
        self.t = None

    @staticmethod
    def set_message(label, message):
        label.value = message
        label.show()

    def get_number_of_selected_features(self):
        return len([2 ** j for j in range(4) if [self._f1_check_box,
                                                 self._f2_check_box,
                                                 self._f3_check_box,
                                                 self._f4_check_box][j].value])

    def _runKarySomAction(self):
        """Button action event"""
        dist_lbl = self.manage_pre_run_gui()
        self.errors_label.hide()

        if self.get_number_of_selected_features() < 1:
            self.set_message(self.errors_label, "At least one feature is required")
            return

        cfg = self.__initialize_cfg(dist_lbl)
        if dist_lbl == WEIGHTED_EUCLIDEAN_DISTANCE and self.get_sum_of_weights() != 100:
            self.set_message(self.errors_label, "Sum of weights must be 100")
            return

        if not os.path.exists(self._input_image_path.value):
            self.set_message(self.errors_label, "Image path {} doesn't exists".format(self._input_image_path.value))
            return
        print("Start processing using cfg:")
        pprint.pprint(cfg)

        self.t = Thread(target=start_process_for_one_image, args=[cfg["input"], cfg["mdist"], "", sum(cfg["features"]),
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
        self.set_message(self.info_label, msg + "\n" + "Start KarySOM for current configuration. Processing...")

    def manage_pre_run_gui(self):
        if self.info_label.visible is True:
            self.info_label.hide()
        if self._dist_label.value.split()[len(self._dist_label.value.split()) - 2] == "Weighted":
            dist_lbl = " ".join(self._dist_label.value.split()[len(self._dist_label.value.split()) - 2:])
        else:
            dist_lbl = self._dist_label.value.split()[len(self._dist_label.value.split()) - 1]
        return dist_lbl

    def __initialize_cfg(self, dist_lbl):
        cfg = {"distance": dist_lbl, "feature_names": [self.FEATURES[i] for i in [j for j in range(3) if
                                                                                  [self._f1_check_box.value,
                                                                                   self._f2_check_box.value,
                                                                                   self._f3_check_box.value,
                                                                                   self._f4_check_box.value][j]]],
               "features": [2 ** j for j in range(4) if [self._f1_check_box,
                                                         self._f2_check_box,
                                                         self._f3_check_box,
                                                         self._f4_check_box][j].value], "epochs": self._epochs_no.value,
               "input": self._input_image_path.value, "pairs": self._pairs_path.value, "rows": self._rows.value,
               "cols": self._cols.value, "weights": dict()}
        if self._f1_check_box.value:
            cfg["weights"][1] = self._f1_w.value / 100
        if self._f2_check_box.value:
            cfg["weights"][2] = self._f2_w.value / 100
        if self._f3_check_box.value:
            cfg["weights"][3] = self._f3_w.value / 100
        if self._f4_check_box.value:
            cfg["weights"][4] = self._f4_w.value / 100

        cfg["mdist"] = False
        if dist_lbl == MANHATTAN_DISTANCE:
            cfg["mdist"] = True
        return cfg

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
        weights_sum = 0
        if self._f1_check_box.value:
            weights_sum += self._f1_w.value
        if self._f2_check_box.value:
            weights_sum += self._f2_w.value
        if self._f3_check_box.value:
            weights_sum += self._f3_w.value
        if self._f4_check_box.value:
            weights_sum += self._f4_w.value
        return weights_sum

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


if __name__ == "__main__":
    from pyforms import start_app

    start_app(KaryML_Main)
