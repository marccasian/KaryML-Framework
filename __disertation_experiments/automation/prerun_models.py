NEW_FEATURES = [3, 5, 6, 7]
FEATURES = [
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
NEW_WEIGHTS = [None, None, None, None]  # SA  - asta
WEIGHTS = [
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

ch_len = 0x01
sh_ch = 0x02
band = 0x04
area = 0x08

features_vals = [ch_len, sh_ch, band, area]

index = 0
last_model = None
we_nr = 0
e_nr = 0
rulat_e = [0 for _ in range(16)]
rulat_we = [0 for _ in range(16)]
for i in range(len(FEATURES)):

    model_name = "E"
    if WEIGHTS[i]:
        model_name = "WE"
        we_nr += 1
    else:
        e_nr += 1

    used_featurs = 0
    weights_lst = []
    idx = 0
    for j in range(len(features_vals)):
        if FEATURES[i] & features_vals[j]:
            used_featurs += features_vals[j]
            if WEIGHTS[i]:
                weights_lst.append(str(WEIGHTS[i][idx]))
                idx += 1
        else:
            if WEIGHTS[i]:
                weights_lst.append("X")
    if WEIGHTS[i]:
        rulat_we[used_featurs] = 1
    else:
        rulat_e[used_featurs] = 1
    model_name += "-{}".format(used_featurs)
    if last_model == model_name:
        index += 1
    else:
        last_model = model_name
        index = 0
    if index > 0:
        model_name += "-{}".format(index)
    print("{} | {}".format(model_name, " | ".join(weights_lst)))
print("Modele cu euclidean {}".format(e_nr))
print("Modele cu weighted euclidean {}".format(we_nr))
print(", ".join([str(i) for i in range(16)][1:]))
print(", ".join([str(i) for i in rulat_e][1:]))
print(", ".join([str(i) for i in rulat_we][1:]))

# nu am rulat cu euclidean features 3,5,6,7
