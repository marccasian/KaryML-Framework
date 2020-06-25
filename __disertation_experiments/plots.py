import os

from __disertation_experiments.dataset.stats import DataSetLoader, CHROMOSOME_LEN_KEY, CHROMOSOME_AREA_KEY, \
    SHORT_CHROMATID_RATIO_KEY

import matplotlib.pyplot as plt


def length_plotter(ds_files):
    dsl = DataSetLoader(ds_files)
    i = 1
    plt.rcdefaults()
    lengths = [int(dsl.ds[i][CHROMOSOME_LEN_KEY] // 5 * 5) for i in dsl.ds]

    x_axis_len_points = list(set(lengths))
    y_axis_len_points = [lengths.count(i) for i in x_axis_len_points]

    plt.bar(x_axis_len_points, y_axis_len_points, align='center', alpha=0.5, width=4)
    plt.xticks([i * 25 for i in range(max(lengths) // 25)])
    plt.yticks([i * 2 for i in range(max(y_axis_len_points) // 2)])
    plt.ylabel('Count')
    plt.xlabel('Chromosome length')
    plt.title('Chromosome length feature distribution')

    plt.show()


def area_plotter(ds_files):
    dsl = DataSetLoader(ds_files)
    i = 1
    plt.rcdefaults()

    areas = [int(dsl.ds[i][CHROMOSOME_AREA_KEY] // 100 * 100) for i in dsl.ds]

    x_axis_area_points = list(set(areas))
    y_axis_area_points = [areas.count(i) for i in x_axis_area_points]

    plt.bar(x_axis_area_points, y_axis_area_points, align='center', alpha=0.5, width=75)
    plt.xticks([i * 1000 for i in range(max(areas) // 1000)])
    plt.yticks([i for i in range(max(y_axis_area_points))])
    plt.ylabel('Count')
    plt.xlabel('Chromosome Area in pixels')
    plt.title('Chromosome area feature distribution')

    plt.show()


def short_ch_ratio_plotter(ds_files):
    dsl = DataSetLoader(ds_files)
    i = 1
    plt.rcdefaults()

    all_values = [int(dsl.ds[i][SHORT_CHROMATID_RATIO_KEY] * 100) / 100 for i in dsl.ds]

    x_axis_area_points = list(set(all_values))
    y_axis_area_points = [all_values.count(i) for i in x_axis_area_points]

    plt.bar(x_axis_area_points, y_axis_area_points, align='center', alpha=0.5, width=0.008)
    plt.xticks([i / 10 for i in range(int((max(all_values) + 0.2) * 10))])
    plt.yticks([i for i in range(max(y_axis_area_points) + 1)])
    plt.ylabel('Count')
    plt.xlabel('Chromosome Short Chromatid Ratio')
    plt.title('Chromosome Short Chromatid Ratio feature distribution')

    plt.show()


def aqe_plotter(aqe_file=r'aqe1.txt'):
    x_points = []
    y_points = []
    with open(aqe_file, 'r') as f:
        for line in f.readlines():
            line = line.strip()
            components = line.split(",")
            if len(components) < 2:
                continue
            x_points.append(int(components[0].strip()))
            y_points.append(float(components[1].strip()))

    fig, ax = plt.subplots()
    ax.plot(x_points, y_points)

    ax.set(xlabel='Epoch', ylabel='AQE Value',
           title='AQE evolution trough training epochs')
    ax.grid()

    fig.savefig(r"__disertation_experiments\plots\{}.png".format(os.path.basename(aqe_file[:-4])))
    plt.show()


if __name__ == '__main__':
    aqe_plotter()
