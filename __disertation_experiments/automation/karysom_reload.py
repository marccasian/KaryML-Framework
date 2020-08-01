import os
import subprocess
import sys

CMD_LINE = r"py -3 __disertation_experiments\automation\runner.py -classifier SOM -ds_entry 1 -features {} -epochs " \
           r"200000 -rows 50 -cols 50"

FEATURES = [0x01, 0x02, 0x04, 0x03, 0x05, 0x06, 0x07]
REMAINING_FEATURES = [0x05, 0x06]
WEIGHTS = [None, None, None, [0.3, 0.7], [0.3, 0.7], [0.3, 0.7], [0.2, 0.2, 0.6]]
REMAINING_WEIGHTS = [[0.3, 0.7], [0.3, 0.7]]

INITIAL_NEURONS_FILE = [
    r"__disertation_experiments\dataset\1\outputs\features_1\2020-06-02_17-51-00\java_neurons.obj",  # PrecKar = 84.153
    r"__disertation_experiments\dataset\1\outputs\features_2\2020-06-02_18-20-10\java_neurons.obj",  # PrecKar = 87.7199
    r"__disertation_experiments\dataset\1\outputs\features_4\2020-06-02_18-53-07\java_neurons.obj",  # PrecKar = 82.7528
    r"__disertation_experiments\dataset\1\outputs\features_3\2020-06-02_19-01-43\java_neurons.obj",  # PrecKar = 82.7765
    r"__disertation_experiments\dataset\1\outputs\features_5\2020-06-02_20-58-36\java_neurons.obj",  # PrecKar = 87.9010
    r"__disertation_experiments\dataset\1\outputs\features_6\2020-06-02_21-46-07\java_neurons.obj",  # PrecKar = 85.1262
    r"__disertation_experiments\dataset\1\outputs\features_7\2020-06-02_19-51-02\java_neurons.obj",  # PrecKar = 87.0179

]
print(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def pre_runs():
    """
    obtained 6 run results for each feature combination in order to obtain a better initial neurons config

    :return:
    """
    times = 1
    total_runs = len(FEATURES) * times
    executed_runs = 0
    for i in range(len(FEATURES)):
        for j in range(times):
            new_cmd_line = CMD_LINE.format(FEATURES[i])
            if WEIGHTS[i]:
                new_cmd_line += " -weights {}".format(" ".join([str(k) for k in WEIGHTS[i]]))
            print("Retry: {};  Running {}".format(j, new_cmd_line))
            os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
            proc = subprocess.Popen(new_cmd_line, stdout=sys.stdout, stderr=sys.stdout)
            sout, stderr = proc.communicate()
            print("Retry: {};  Executed {}. Stdout:{}, Stderr: {}".format(j, new_cmd_line, sout, stderr))
            executed_runs += 1
            print("Progress: {}%. {} remaining runs".format((executed_runs / total_runs) * 100,
                                                            total_runs - executed_runs))


if __name__ == '__main__':
    pre_runs()
