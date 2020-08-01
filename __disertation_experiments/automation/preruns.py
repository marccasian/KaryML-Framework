import os
import random
import subprocess
import sys
import time

CMD_LINE = r"py -3 __disertation_experiments\automation\runner_remainings.py -classifier SOM -ds_entry {} -features " \
           r"{} -epochs 200000 -rows 25 -cols 25"

# L -ch len                         1
# S - short chromatid ratio         2
# B - banding profile               4
# A - ch area                       8

NEW_FEATURES = [1]
FEATURES = [
    3, 5, 6, 7,
    1,  # L
    2,  # S
    4,  # B
    8,  # A
    3,  # LS
    5,  # LB
    6,  # SB
    9,  # LA
    9,  # LA
    9,  # LA
    9,  # LA
    9,  # LA
    10,  # SA
    10,  # SA
    10,  # SA
    10,  # SA
    10,  # SA
    12,  # BA
    12,  # BA
    12,  # BA
    12,  # BA
    12,  # BA
    7,  # LSB
    11,  # LSA
    11,  # LSA
    11,  # LSA
    11,  # LSA
    13,  # LBA
    13,  # LBA
    13,  # LBA
    13,  # LBA
    13,  # LBA
    14,  # SBA
    14,  # SBA
    14,  # SBA
    14,  # SBA
    15,  # LSBA
    15,  # LSBA
    15,  # LSBA
    15,  # LSBA
    15,  # LSBA
]
NEW_WEIGHTS = [None]
WEIGHTS = [
    None, None, None, None,
    None,  # L
    None,  # S
    None,  # B
    None,  # A
    [0.3, 0.7],  # LS
    [0.3, 0.7],  # LB
    [0.3, 0.7],  # SB
    None,  # LA
    [0.6, 0.4],  # LA
    [0.4, 0.6],  # LA
    [0.3, 0.7],  # LA
    [0.7, 0.3],  # LA
    None,  # SA
    [0.4, 0.6],  # SA
    [0.6, 0.4],  # SA
    [0.3, 0.7],  # SA
    [0.7, 0.3],  # SA  - asta
    None,  # BA
    [0.6, 0.4],  # BA
    [0.4, 0.6],  # BA
    [0.7, 0.3],  # BA
    [0.3, 0.7],  # BA
    [0.2, 0.2, 0.6],  # LSB
    None,  # LSA
    [0.2, 0.2, 0.6],  # LSA
    [0.2, 0.6, 0.2],  # LSA
    [0.6, 0.2, 0.2],  # LSA
    None,  # LBA
    [0.15, 0.7, 0.15],  # LBA
    [0.1, 0.8, 0.1],  # LBA
    [0.2, 0.6, 0.2],  # LBA
    [0.3, 0.4, 0.3],  # LBA
    None,  # SBA
    [0.2, 0.2, 0.6],  # SBA
    [0.2, 0.6, 0.2],  # SBA
    [0.6, 0.2, 0.2],  # SBA
    None,  # LSBA
    [0.1, 0.2, 0.6, 0.1],  # LSBA
    [0.2, 0.1, 0.3, 0.2],  # LSBA
    [0.2, 0.2, 0.4, 0.2],  # LSBA
    [0.3, 0.1, 0.3, 0.3],  # LSBA
]


SELECTED_FEATURES = [
    3,  # Model 0 used features 3 has avg PrecKar 89.71865671863357
    13,  # Model 33 used features 13 weights 0.1 0.8 0.1 has avg PrecKar 89.40693687939358
    10,  # Model 18 used features 10 weights 0.6 0.4 has avg PrecKar 89.40092405029525
    11,  # Model 29 used features 11 weights 0.2 0.6 0.2 has avg PrecKar 89.31775115622007
    15,  # Model 44 used features 15 weights 0.3 0.1 0.3 0.3 has avg PrecKar 89.14354638474389
    9,  # Model 12 used features 9 weights 0.6 0.4 has avg PrecKar 89.12659028309345
    1,  # Model 4 used features 1 has avg PrecKar 89.1255662622695
    3,  # Model 8 used features 3 weights 0.3 0.7 has avg PrecKar 88.03329895634404
    7,  # Model 26 used features 7 weights 0.2 0.2 0.6 has avg PrecKar 87.98756203583098
    8,  # Model 7 used features 8 has avg PrecKar 87.87030055118304
    12,  # Model 21 used features 12 has avg PrecKar 87.48883786803303
    5,  # Model 9 used features 5 weights 0.3 0.7 has avg PrecKar 87.21327174021701
    14,  # Model 36 used features 14 has avg PrecKar 87.2078616658087
    4,  # Model 6 used features 4 has avg PrecKar 87.07253639531639
    2,  # Model 5 used features 2 has avg PrecKar 86.4099875142842
    6,  # Model 10 used features 6 weights 0.3 0.7 has avg PrecKar 85.66436499517106
]

SELECTED_WEIGHTS = [
    None,  # Model 0 used features 3 has avg PrecKar 89.71865671863357
    [0.1, 0.8, 0.1],  # Model 33 used features 13 weights 0.1 0.8 0.1 has avg PrecKar 89.40693687939358
    [0.6, 0.4],  # Model 18 used features 10 weights 0.6 0.4 has avg PrecKar 89.40092405029525
    [0.2, 0.6, 0.2],  # Model 29 used features 11 weights 0.2 0.6 0.2 has avg PrecKar 89.31775115622007
    [0.3, 0.1, 0.3, 0.3],  # Model 44 used features 15 weights 0.3 0.1 0.3 0.3 has avg PrecKar 89.14354638474389
    [0.6, 0.4],  # Model 12 used features 9 weights 0.6 0.4 has avg PrecKar 89.12659028309345
    None,  # Model 4 used features 1 has avg PrecKar 89.1255662622695
    [0.3, 0.7],  # Model 8 used features 3 weights 0.3 0.7 has avg PrecKar 88.03329895634404
    [0.2, 0.2, 0.6],  # Model 26 used features 7 weights 0.2 0.2 0.6 has avg PrecKar 87.98756203583098
    None,  # Model 21 used features 12 has avg PrecKar 87.48883786803303
    [0.3, 0.7],  # Model 9 used features 5 weights 0.3 0.7 has avg PrecKar 87.21327174021701
    None,  # Model 36 used features 14 has avg PrecKar 87.2078616658087
    None,  # Model 6 used features 4 has avg PrecKar 87.07253639531639
    None,  # Model 5 used features 2 has avg PrecKar 86.4099875142842
    [0.3, 0.7]  # Model 10 used features 6 weights 0.3 0.7 has avg PrecKar 85.66436499517106
]

print(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def pre_runs():
    """
    obtained 6 run results for each feature combination in order to obtain a better initial neurons config

    :return:
    """
    assert len(WEIGHTS) == len(FEATURES)
    times = 1
    ds_entries = [1, 3, 5, 6, 7, 8]
    feature_indexes = [i for i in range(len(FEATURES))]
    times_indexes = [i for i in range(times)]
    random.shuffle(ds_entries)
    random.shuffle(feature_indexes)
    random.shuffle(times_indexes)
    total_runs = len(FEATURES) * times * len(ds_entries)
    executed_runs = 0
    for i in feature_indexes:
        for ds_entry in ds_entries:
            for j in times_indexes:
                new_cmd_line = CMD_LINE.format(ds_entry, FEATURES[i])
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
