import datetime
import json
import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from F_Experiments_Helper import db
from F_Experiments_Helper.run_instance import RunInstance
from __disertation_experiments.automation import runner
from __disertation_experiments.automation.get_best_initial_neurons_for_each_model import DIM, ds_entries, OVER_ID, \
    ORDERED_SELECTED_FEATURES, ORDERED_SELECTED_WEIGHTS, MODELS_LABEL
from C_FeatureExtraction.feature_extractions_constants import *

ylim = [75, 90]
tick_val = 2

euclidean_distance = {
    1: [
        81.0762444,
        81.2171522,
        81.3145117,
        81.6721359,
        81.7362524,
        81.7362524,
        82.0344302,
        82.2568571,
        82.4876462,
        82.6287609,
        82.9083251,
        83.2795306,
        83.3209563,
        83.4037149,
        83.5927973,
        83.9042164,
        83.9774479,
        84.0793105,
        84.3691959,
        84.5760428,
        85.0900216,
        85.1220503,
        85.3630003,
        85.5044019,
        85.6746887,
        85.6833083,
        85.7281846,
        85.9617371,
        86.0745237,
        86.2590914,
        86.3281521,
        86.4081789,
        86.4169111,
        86.5025598,
        86.5636111,
        86.7424659,
        86.7598034,
        86.8271198,
        86.8867933,
        87.0421144,
        87.5380941
    ],
    2: [
        80.4044743,
        80.4426485,
        80.7020432,
        80.7886877,
        81.1006353,
        81.2244395,
        81.2268623,
        81.4702969,
        81.5683882,
        81.6319690,
        81.7515908,
        81.8576991,
        81.9063727,
        81.9513339,
        82.1872587,
        82.5567222,
        82.7517378,
        82.8234694,
        83.4972327,
        84.0750143,
        84.8309108,
        85.0133640,
        85.0696151,
        85.3841025
    ],
    3: [
        81.9434268,
        82.2179409,
        82.3690453,
        82.5331395,
        82.6206124,
        83.4657569,
        83.5817536,
        84.0140792,
        84.0159827,
        84.1518148,
        84.2529884,
        84.2857464,
        84.4240714,
        84.8571768,
        84.8780118,
        84.8797170,
        84.9191451,
        85.0038116,
        85.1577788,
        85.3295425,
        85.4133946,
        85.4688760,
        85.6541772,
        85.8078337,
        85.9435755,
        86.3064365,
        86.3841105,
        86.8439238,
        86.8639353,
        86.9646387,
        86.9837187,
        87.2226782,
        87.2403929,
        87.6777069,
        88.5073099,
        88.5521503
    ],
    4: [
        80.7249018,
        81.0591641,
        82.6871946,
        83.0666871,
        83.4073785,
        83.8718436,
        83.9324176,
        83.9489537,
        83.9693797,
        84.7253639,
        84.7740904,
        84.8494134,
        85.0387135,
        85.2942620,
        85.3116668,
        85.3116668,
        85.3241149,
        85.3878406,
        85.3888175,
        86.1777341,
        86.2546639,
        86.8298331,
        86.8439057,
        86.9980200,
        87.0520376,
        87.1493269,
        87.5244422,
        87.6998511,
        87.8427966,
        88.3421409,
        89.2602477
    ],
    5: [
        80.4018379,
        81.9745657,
        82.9285259,
        83.1858302,
        83.6786535,
        83.8000843,
        83.8299860,
        84.8895067,
        84.8899674,
        84.9419354,
        84.9570049,
        85.1443995,
        85.5546344,
        85.5947328,
        85.9812305,
        86.1373796,
        86.8205153,
        86.9417081,
        87.0442304,
        87.2135362,
        87.3118050,
        87.4638698,
        87.6934673,
        87.8657308,
        87.9250058,
        87.9681027,
        87.9795155,
        88.0665467,
        88.1184535,
        88.1700763,
        88.5252303,
        88.6830524,
        88.6979559,
        88.7980254,
        89.1984068,
        89.4096906,
        89.4981502,
        89.5041482,
        89.5073625,
        89.7266351,
        89.8593733,
        89.9843346,
        90.2523148,
        90.3622600,
        90.6006547,
        90.6267603
    ],
    6: [
        80.3085431,
        80.8606363,
        81.4865078,
        82.2932760,
        82.3565215,
        82.6646679,
        82.7660137,
        82.7878316,
        83.4680651,
        83.5333884,
        83.6382237,
        83.9698746,
        84.0407157,
        84.1098928,
        84.2591643,
        84.2879302,
        84.5395238,
        84.6318986,
        85.0604976,
        85.2669678,
        85.3104457,
        85.6358788,
        85.7500586,
        85.8636757,
        85.8896402,
        86.0442204,
        86.0551326,
        86.3895787,
        86.8694817,
        86.8940674,
        87.1561167,
        87.4918790,
        87.7028924,
        87.9174724,
        87.9290707,
        87.9960999,
        88.1777774,
        88.1914809,
        88.2952478,
        88.3864859,
        88.4728235
    ],
    7: [
        80.0041577,
        80.0613716,
        81.7031263,
        81.8419986,
        81.8625973,
        82.1316541,
        82.3435556,
        82.5280149,
        82.6449914,
        83.3327233,
        83.6624707,
        83.8641269,
        84.0775128,
        84.2211303,
        84.3440466,
        84.7894949,
        84.9890854,
        85.2523972,
        85.4773374,
        85.5803064,
        85.9304593,
        86.5307093,
        86.7198089,
        86.9681633,
        87.0628709,
        87.0974619,
        87.1828316,
        87.1897351,
        87.2943381,
        87.3120946,
        87.3835750,
        87.4747138,
        87.5796580,
        88.1917791,
        88.2953636,
        88.3245452,
        88.4035095,
        88.7715279,
        88.9963369,
        89.3650883
    ]
}

w_euclidean_distance = {
    4: [
        80.4783062,
        80.5172703,
        80.6874880,
        80.6905712,
        80.7608711,
        81.0349438,
        81.2006150,
        82.0613676,
        82.1912170,
        82.5323739,
        82.5967089,
        82.6208513,
        82.7607947,
        82.7657263,
        83.0129365,
        83.0717507,
        84.2916357,
        84.4603180,
        84.7720488,
        84.9528262,
        85.0558366,
        85.0595846,
        85.4167890,
        85.5161574,
        85.5351842,
        85.7759963,
        85.8046117,
        85.8513853,
        85.9384783,
        86.1677441,
        86.3150835,
        86.6155945,
        87.7005035,
        88.2232800

    ],
    5: [
        84.3033468,
        84.4310023,
        84.7079318,
        85.0957591,
        85.1200121,
        85.2648660,
        85.2767097,
        85.2972709,
        85.5714409,
        85.6274827,
        86.0633339,
        86.1005875,
        86.3054479,
        86.5191217,
        86.8121028,
        86.8823202,
        86.9835468,
        87.2976959,
        87.4084507,
        87.5034349,
        87.5396210,
        87.9124356,
        87.9143999,
        87.9267208,
        88.1570946,
        88.2619305,
        88.3121320,
        88.3472482,
        88.3785550,
        88.6870344,
        88.6924570,
        88.9101172,
        88.9209079,
        88.9940281,
        89.6537242,
        89.9013926,
        90.9177864,
        90.9397582,

    ],
    6: [
        81.8697915,
        81.8783093,
        81.8895692,
        81.9817768,
        82.0734197,
        82.2743769,
        82.2882039,
        82.3098981,
        82.3151134,
        82.8197754,
        82.9943702,
        83.1207350,
        83.2587335,
        83.5444491,
        83.6654038,
        83.8383919,
        84.1209848,
        84.1397316,
        84.2849873,
        84.5753425,
        84.9842650,
        85.1316089,
        85.6629984,
        86.2411349,
        86.5126887,
        87.3737752,
        87.4474625,
        87.5107456,
        87.7753520,
        88.0670541,
        88.2242595,
        88.8057614,
        88.8799283,
        89.4492005,
        90.0071071,
        90.1041787,

    ],
    7: [
        84.7708143,
        84.8617450,
        84.9537067,
        85.2702740,
        85.3485914,
        85.3621905,
        85.3786639,
        85.5275566,
        85.5680513,
        85.7990557,
        85.8588786,
        86.0687567,
        86.3037033,
        86.3735286,
        86.5465304,
        86.9594380,
        87.3838169,
        87.4858527,
        87.5087617,
        87.5551640,
        87.6060487,
        87.7223119,
        87.7644866,
        87.9272833,
        87.9604611,
        88.0168238,
        88.2408958,
        88.2610089,
        88.5264571,
        88.6171798,
        89.2496180,

    ]
}

manhattan_distance = {
    1: [
        80.4605244,
        80.9361521,
        81.1256262,
        81.2253309,
        82.6823773,
        82.7645219,
        83.2868042,
        83.2940217,
        84.4272212,
        84.5867941,
        84.8488258,
        85.0263312,
        85.0548777,
        85.9421648,
        86.0964264,
        86.2242649,
        86.7127565,
        86.9471801,
        87.8134713,
        88.8971850,
        89.3388285,

    ],
    2: [
        80.0154571,
        80.0777409,
        80.2664630,
        80.4520427,
        80.6363846,
        81.2429959,
        81.4804480,
        81.9844428,
        82.8880867,
        82.9805562,
        83.1303582,
        83.1649599,
        83.6347031,
        83.7474124,
        83.7579396,
        84.1383412,
        84.1731320,
        84.9066452,
        85.0635390,
        85.2154360,
        85.3964724,
        86.3860311,
        87.3599546,

    ],
    3: [
        81.3412674,
        81.7132322,
        81.8545675,
        81.9107052,
        82.3192321,
        82.4248203,
        82.4303598,
        82.4889957,
        82.5908793,
        83.8075767,
        83.9005835,
        84.2285010,
        84.6950449,
        85.7692074,
        85.8667607,
        86.2963647,
        86.2979663,
        86.9610841,
        87.3182486,
        87.4180309,
        88.1560164,
        89.7022322,
        89.9972801,

    ],
    4: [
        80.9356617,
        81.1353107,
        81.3679537,
        81.8185434,
        82.1154247,
        82.2509672,
        82.2794033,
        82.3731447,
        82.6009571,
        83.1927944,
        83.2037203,
        83.8941553,
        83.8941553,
        83.9408422,
        84.1833582,
        84.1857127,
        85.1492317,
        85.1571492,
        85.2545859,
        85.4283425,
        86.8808082,
        87.1490372,
        87.3997028,
        88.0206387,
        88.0918111,
        88.9001980,
    ],
    5: [
        81.0133927,
        81.1001270,
        81.2871801,
        82.9954658,
        83.1881439,
        83.6677588,
        83.9138178,
        84.6032686,
        84.6431671,
        85.3581169,
        85.4523784,
        85.6036277,
        85.8893273,
        86.4320705,
        87.2106253,
        87.2211674,
        87.3654570,
        87.5014001,
        88.5593738,
        89.8813056,
        90.6104731,

    ],
    6: [
        81.1777574,
        81.9329459,
        82.0377041,
        82.9664143,
        83.3891863,
        84.0460710,
        84.3088658,
        84.4506041,
        84.4791197,
        84.6573685,
        85.1727423,
        85.4358126,
        85.6953697,
        85.7246328,
        85.7420836,
        86.3664870,
        86.4287046,
        86.7136144,
        86.8217477,
        87.3792596,
        87.4149918,
    ],
    7: [
        80.5781148,
        80.6524665,
        81.5889953,
        82.1213844,
        82.1408027,
        82.6872124,
        82.9345740,
        83.3245571,
        83.9253001,
        84.3444553,
        84.3581706,
        84.4303022,
        84.4712834,
        85.0202677,
        85.5616112,
        85.7897519,
        86.4344343,
        87.4219043,
        88.2138971,
        88.3286441,
        88.3556990,
        88.9185408,
        89.4711208,

    ]
}


def list_avg(l):
    return sum(l) / len(l)


def build_plot():
    e_m = [_ for _ in range(len(euclidean_distance))]
    e_std = [_ for _ in range(len(euclidean_distance))]
    e_stderr = [_ for _ in range(len(euclidean_distance))]

    for i in range(1, len(euclidean_distance) + 1):
        e_m[i - 1] = np.mean(euclidean_distance[i])
        e_std[i - 1] = np.std(euclidean_distance[i])
        e_stderr[i - 1] = np.std(euclidean_distance[i]) / np.sqrt(len(euclidean_distance[i]))

    w_e_m = [_ for _ in range(len(w_euclidean_distance))]
    w_e_std = [_ for _ in range(len(w_euclidean_distance))]
    w_e_stderr = [_ for _ in range(len(w_euclidean_distance))]

    for i in range(4, len(w_euclidean_distance) + 4):
        w_e_m[i - 4] = np.mean(w_euclidean_distance[i])
        w_e_std[i - 4] = np.std(w_euclidean_distance[i])
        w_e_stderr[i - 4] = np.std(w_euclidean_distance[i]) / np.sqrt(len(w_euclidean_distance[i]))

    m_m = [_ for _ in range(len(manhattan_distance))]
    m_std = [_ for _ in range(len(manhattan_distance))]
    m_stderr = [_ for _ in range(len(manhattan_distance))]

    for i in range(1, len(manhattan_distance) + 1):
        m_m[i - 1] = np.mean(manhattan_distance[i])
        m_std[i - 1] = np.std(manhattan_distance[i])
        m_stderr[i - 1] = np.std(manhattan_distance[i]) / np.sqrt(len(manhattan_distance[i]))

    gs = gridspec.GridSpec(1)
    fig = plt.figure(figsize=(18, 6))

    font = {'family': 'normal',
            'weight': 'bold',
            'size': 20}

    plt.rc('font', **font)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_title("Euclidean Distance", font)
    # ax1.set_xlabel("Model", font)
    ax1.set_ylabel("PrecKar", font)
    ax1.bar(["M" + str(i) for i in range(1, len(e_m) + 1)], [e_m[i] for i in range(len(e_m))], align='center')
    for i in range(len(e_m)):
        stderr = np.std(euclidean_distance[i + 1]) / np.sqrt(len(euclidean_distance[i + 1]))
        print("ed-M%d -mean = %s" % (i, str(np.mean(euclidean_distance[i + 1]))))
        print("ed-M%d -std = %s" % (i, str(np.std(euclidean_distance[i + 1]))))
        print("ed-M%d -stderr = %s" % (i, str(stderr)))
        conf = 1.96 * stderr
        print("ed-M%d -conf = %s" % (i, str(conf)))
        ax1.errorbar(i, e_m[i], yerr=conf, ecolor="r")
    start, end = ax1.get_ylim()
    ax1.yaxis.set_ticks(np.arange(start, end, tick_val))
    ax1.set_ylim(ylim)

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_xlabel("Model", font)
    # ax2.set_ylabel("PrecKar", font)
    ax2.set_title("Weighted Euclidean Distance", font)
    ax2.bar(["M" + str(i) for i in range(4, len(w_e_m) + 4)], [w_e_m[i] for i in range(len(w_e_m))], align='center')
    for i in range(len(w_e_m)):
        stderr = np.std(w_euclidean_distance[i + 4]) / np.sqrt(len(w_euclidean_distance[i + 4]))
        print("wed-M%d -mean = %s" % (i, str(np.mean(w_euclidean_distance[i + 4]))))
        print("wed-M%d -std = %s" % (i, str(np.std(w_euclidean_distance[i + 4]))))
        print("wed-M%d -stderr = %s" % (i, str(stderr)))
        conf = 1.96 * stderr
        print("wed-M%d -conf = %s" % (i, str(conf)))
        ax2.errorbar(i, w_e_m[i], yerr=conf, ecolor="r")
    start, end = ax1.get_ylim()
    ax2.yaxis.set_ticks(np.arange(start, end, tick_val))
    ax2.set_ylim(ylim)

    ax3 = fig.add_subplot(gs[0, 2])
    # ax3.set_xlabel("Model",font)
    # ax3.set_ylabel("PrecKar", font)
    ax3.set_title("Manhattan Distance", font)
    ax3.bar(["M" + str(i) for i in range(1, len(m_m) + 1)], [m_m[i] for i in range(len(m_m))], align='center')
    for i in range(len(m_m)):
        stderr = np.std(manhattan_distance[i + 1]) / np.sqrt(len(manhattan_distance[i + 1]))
        print("m-M%d -mean = %s" % (i, str(np.mean(manhattan_distance[i + 1]))))
        print("m-M%d -std = %s" % (i, str(np.std(manhattan_distance[i + 1]))))
        print("m-M%d -stderr = %s" % (i, str(stderr)))
        conf = 1.96 * stderr
        print("m-M%d -conf = %s" % (i, str(conf)))
        ax3.errorbar(i, m_m[i], yerr=conf, ecolor="r")
    start, end = ax1.get_ylim()
    ax3.yaxis.set_ticks(np.arange(start, end, tick_val))
    ax3.set_ylim(ylim)

    ax1.grid(color='gray', linestyle=':', linewidth=1)
    ax2.grid(color='gray', linestyle=':', linewidth=1)
    ax3.grid(color='gray', linestyle=':', linewidth=1)

    plt.savefig("grafic%s.png" % datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    plt.show()


def new_build_plot(classifier):
    info = get_best_initial_input_neurons_file(classifier)
    yellow = "#f8ff00"
    orange = "#f8a102"
    blue = "#34cdf9"
    red = "#f8d9d5"
    font = {'family': 'normal',
            'weight': 'bold',
            'size': 16}
    fig = plt.figure(figsize=(18, 6))
    plt.rc('font', **font)
    plt.xlabel("Model", font)
    plt.ylabel("PrecKar", font)
    barlist = plt.bar(MODELS_LABEL, [i[0] for i in info], align="center", zorder=3)
    for i in range(len(barlist)):
        plt.errorbar(i, info[i][0], yerr=info[i][1], ecolor="r", zorder=5)
        barlist[i].set_color(red)
    barlist[0].set_color(yellow)
    barlist[7].set_color(yellow)
    barlist[8].set_color(yellow)
    barlist[9].set_color(blue)
    barlist[13].set_color(yellow)
    barlist[15].set_color(yellow)
    if classifier == "SOM":
        barlist[11].set_color(orange)
    fig.autofmt_xdate()
    plt.grid(color='gray', linestyle=':', linewidth=1, zorder=0)
    start, end = plt.ylim()
    plt.yticks = np.arange(start, end, tick_val)
    plt.ylim(ylim)
    plt.title(
        "PrecKar values obtained during {} Scoring Runs ".format(classifier), fontdict=font)

    plt.show()


def get_best_initial_input_neurons_file(classifier_type="SOM"):
    if classifier_type == "FSOM":
        db.DB_FILE = r'__disertation_experiments\karypy_runs_eval_checkpoint_30_fsom_runs.db'
    feature_indexes = [i for i in range(len(ORDERED_SELECTED_FEATURES))]
    total_models = len(ORDERED_SELECTED_FEATURES) * len(ds_entries)
    with open(os.path.join(runner.DS_ROOT_DIR_PATH, "description.json"), "r") as f:
        ds_descriptor = json.load(f)
    print("Searching for best run for {} models".format(total_models))
    dss = runner.ClassifierReqFilesGenerator(ds_descriptor)
    best_runs_for_models = []
    to_ret = list()
    for i in feature_indexes:
        runs = list()
        for ds_entry in ds_entries:
            input_image_path = ds_descriptor[str(ds_entry)][db.INPUT_IMAGE_PATH_COLUMN_NAME]
            root_dir_path, features_file_path, features_weight_file_path, feature_used_list, \
            outputs_dir_path = dss.generate_input_feature_files(
                str(ds_entry), ORDERED_SELECTED_FEATURES[i], ORDERED_SELECTED_WEIGHTS[i])
            a = RunInstance(classifier_type=classifier_type, input_image_path=input_image_path,
                            features_file_path=features_file_path, used_features=ORDERED_SELECTED_FEATURES[i],
                            distance=runner.determine_distance_type(False, ORDERED_SELECTED_WEIGHTS[i]), epochs=200000,
                            rows=DIM[classifier_type], cols=DIM[classifier_type],
                            len_feature_weight=runner.get_feature_weight(feature_used_list, CHROMOSOME_LEN_KEY),
                            short_chromatid_ratio_feature_weight=runner.get_feature_weight(feature_used_list,
                                                                                           SHORT_CHROMATID_RATIO_KEY),
                            banding_pattern_feature_weights=runner.get_feature_weight(feature_used_list,
                                                                                      BANDAGE_PROFILE_KEY),
                            area_feature_weight=runner.get_feature_weight(feature_used_list, CHROMOSOME_AREA_KEY))
            runs.extend(db.get_similar_runs(a, over_id=OVER_ID)[:25])
        runs_preckar = [j.preckar for j in runs]
        stderr = np.std(runs_preckar) / np.sqrt(len(runs_preckar))
        if ORDERED_SELECTED_WEIGHTS[i]:
            print("Model {}; Features: {}, Weights: {}".format(i, ORDERED_SELECTED_FEATURES[i],
                                                               ", ".join(
                                                                   str(ll) for ll in ORDERED_SELECTED_WEIGHTS[i])))
        else:
            print("Model {}; Features: {}".format(i, ORDERED_SELECTED_FEATURES[i]))
        print("\t Best PrecKar = {:.2f}%".format(np.max(runs_preckar)))
        # print("\t Mean = {:.2f}%".format(np.mean(runs_preckar)))
        # print("\t STD = {}".format(np.std(runs_preckar)))
        # print("\t STDERR = {}".format(stderr))
        conf = 1.96 * stderr
        # print("\t CI = {:.2f}%".format(conf))
        print("\t AVG +- CI = {:.2f}+-{:.2f}%".format(np.mean(runs_preckar), conf))
        to_ret.append([np.mean(runs_preckar), conf])
    return to_ret


def kary_ml_vs_karysom(classifier_type="SOM"):
    karysom_info = [
        [84.537, 0.59],  # Euclidean M1 = E-1
        [82.342, 0.6],  # Euclidean M2 = E-2
        [85.356, 0.69],  # Euclidean M4 = E-3
        [83.895, 0.73],  # Weighted Euclidean M4 = WE-3
        [85.187, 0.56],  # Euclidean M3 = E-4
        [87.287, 0.54],  # Weighted Euclidean M5 = WE-5
        [85.039, 0.85],  # Weighted Euclidean M6 = WE-6
        [86.799, 0.45],  # Weighted Euclidean M7 = WE-7
    ]

    NEW_MODELS_LABEL = [
        "E-1|M1",
        "E-2|M2",
        "E-3|M4",
        "WE-3|W-M4",
        "E-4|M3",
        "WE-5|M5",
        "WE-6|M6",
        "WE-7|M7",
        "WE-9-0",
    ]

    feature_indexes = [0, 1, 2, 3, 4, 5, 6, 7]
    with open(os.path.join(runner.DS_ROOT_DIR_PATH, "description.json"), "r") as f:
        ds_descriptor = json.load(f)
    dss = runner.ClassifierReqFilesGenerator(ds_descriptor)
    kary_ml_info = list()
    for i in feature_indexes:
        runs = list()
        for ds_entry in ds_entries:
            input_image_path = ds_descriptor[str(ds_entry)][db.INPUT_IMAGE_PATH_COLUMN_NAME]
            root_dir_path, features_file_path, features_weight_file_path, feature_used_list, \
            outputs_dir_path = dss.generate_input_feature_files(
                str(ds_entry), ORDERED_SELECTED_FEATURES[i], ORDERED_SELECTED_WEIGHTS[i])
            a = RunInstance(classifier_type=classifier_type, input_image_path=input_image_path,
                            features_file_path=features_file_path, used_features=ORDERED_SELECTED_FEATURES[i],
                            distance=runner.determine_distance_type(False, ORDERED_SELECTED_WEIGHTS[i]), epochs=200000,
                            rows=DIM[classifier_type], cols=DIM[classifier_type],
                            len_feature_weight=runner.get_feature_weight(feature_used_list, CHROMOSOME_LEN_KEY),
                            short_chromatid_ratio_feature_weight=runner.get_feature_weight(feature_used_list,
                                                                                           SHORT_CHROMATID_RATIO_KEY),
                            banding_pattern_feature_weights=runner.get_feature_weight(feature_used_list,
                                                                                      BANDAGE_PROFILE_KEY),
                            area_feature_weight=runner.get_feature_weight(feature_used_list, CHROMOSOME_AREA_KEY))
            runs.extend(db.get_similar_runs(a, over_id=OVER_ID)[:25])
        runs_preckar = [j.preckar for j in runs]
        stderr = np.std(runs_preckar) / np.sqrt(len(runs_preckar))
        conf = 1.96 * stderr
        kary_ml_info.append([np.mean(runs_preckar), conf])
    width = 0.25

    r1 = np.arange(len(karysom_info))
    r2 = [x + width for x in r1]

    yellow = "#f8ff00"
    orange = "#f8a102"
    blue = "#34cdf9"
    red = "#f8d9d5"
    font = {'family': 'normal',
            'weight': 'bold',
            'size': 16}
    fig = plt.figure(figsize=(15, 7))
    plt.rc('font', **font)
    plt.xlabel("Model", font)
    plt.ylabel("PrecKar", font)
    new_r1 = list(r1)
    new_r1.append(r1[-1]+1)
    new_kary_ml_info = kary_ml_info[:]
    new_kary_ml_info.append([86.03, 0.33])
    # plt.bar(r1[-1] + 1, 86.03, zorder=3, edgecolor='white', label='KaryML Framework')

    # plt.errorbar(r1[-1] + 1, 86.03, yerr=0.33, ecolor="r", zorder=5)
    karysom_barlist = plt.bar(r2, [i[0] for i in karysom_info], zorder=3, edgecolor='white', label='KarySOM', width=width)
    for i in range(len(karysom_barlist)):
        plt.errorbar(r2[i], karysom_info[i][0], yerr=karysom_info[i][1], ecolor="r", zorder=5)

    kary_ml_barlist = plt.bar(new_r1, [i[0] for i in new_kary_ml_info], zorder=3, edgecolor='white', label='KaryML Framework', width=width)
    for i in range(len(kary_ml_barlist)):
        plt.errorbar(new_r1[i], new_kary_ml_info[i][0], yerr=new_kary_ml_info[i][1], ecolor="r", zorder=5)
        # barlist[i].set_color(red)
    # barlist[0].set_color(yellow)
    # barlist[7].set_color(yellow)
    # barlist[8].set_color(yellow)
    # barlist[9].set_color(blue)
    # barlist[13].set_color(yellow)
    # barlist[15].set_color(yellow)
    plt.xticks([r + width for r in range(len(new_kary_ml_info))], NEW_MODELS_LABEL)
    fig.autofmt_xdate()
    plt.grid(color='gray', linestyle=':', linewidth=1, zorder=0)
    start, end = plt.ylim()
    plt.yticks = np.arange(start, end, tick_val)
    plt.ylim(ylim)
    plt.title("PrecKar values KaryML(SOM Models) vs KarySOM", fontdict=font)
    plt.legend()

    plt.show()


if __name__ == '__main__':
    # new_build_plot("SOM")
    # get_best_initial_input_neurons_file("SOM")
    kary_ml_vs_karysom()
