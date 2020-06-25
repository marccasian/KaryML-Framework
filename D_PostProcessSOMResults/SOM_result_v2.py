import math
import os
import traceback

import D_PostProcessSOMResults.SOM_result_loader as SOM_result_loader
import D_PostProcessSOMResults.SOM_result_neuron_entry as SOM_result_neuron_entry
from D_PostProcessSOMResults.SOM_results_constants import *
from D_PostProcessSOMResults.expected_karyotype_loader import ExpectedKaryotype


class SOMResultsV2:
    def __init__(self, results_file_path):
        self.results_file_path = results_file_path
        self.__dist_matrix_file = results_file_path[:-4] + DIST_MATRIX_FILE_SUFFIX
        self.result_loader = SOM_result_loader.SOMResultsLoader(results_file_path)
        self.som_chromosomes, self.som_neurons = self.result_loader.load_results()
        self.number_of_chromosomes = self.result_loader.number_of_chromosomes
        self.number_of_neurons = self.result_loader.number_of_neurons
        self.extended_map = list()
        self.full_map = list()
        self.ch_dist_matrix = list()
        self.karyotype = list()

    def pretty_print_som_results(self):
        print("Chromosomes:")
        for entry in self.som_chromosomes:
            print("\tID: %s" % entry.ch_id)
            print("\t\tColor code: [R:%s G:%s B:%s]" % (entry.rgb_tuple[0], entry.rgb_tuple[1], entry.rgb_tuple[2]))
            print("\t\tImage path: %s" % entry.ch_img_path)
            print("\t\tSOM Torus Coordinates: [X:%s Y:%s]" % (entry.x, entry.y))
        print("Neurons:")
        for entry in self.som_neurons:
            print("\t%s" % entry)

    def __compute_extended_map(self):
        """
            Extended map is a map used to compute distance between 2 neurons
            Shape 3*som_height X 3* som_width
            Each [i*som_height:(i+1)som_height-1 X j*som_width:(j+1)*som_width] is equal to full map , i,j = 1..3

        :return: None, but self.extended_map is generated
        """
        points_matrix = [
            [(-1, -1), (-1, 0), (-1, 1)],
            [(0, -1), (0, 0), (0, 1)],
            [(1, -1), (1, 0), (1, 1)]
        ]
        self.extended_map = [
            [
                None
                for _ in range(len(points_matrix[0]) * self.result_loader.som_matrix_width)
            ]
            for _ in range(len(points_matrix) * self.result_loader.som_matrix_height)
        ]

        for i in range(len(self.extended_map)):
            for j in range(len(self.extended_map[0])):
                normalized_i_neuron_index = i % self.result_loader.som_matrix_height
                normalized_j_neuron_index = j % self.result_loader.som_matrix_width
                i_sign_index = i // self.result_loader.som_matrix_height
                j_sign_index = j // self.result_loader.som_matrix_width
                current_neuron = self.full_map[normalized_i_neuron_index][normalized_j_neuron_index]
                new_x = current_neuron.x + (points_matrix[i_sign_index][j_sign_index][0]) * self.result_loader.som_width
                new_y = current_neuron.y + (
                    points_matrix[i_sign_index][j_sign_index][
                        0]) * self.result_loader.som_height
                new_neuron = SOM_result_neuron_entry.SOMNeuronResultEntry(
                    current_neuron.neuron_id,
                    i,
                    j,
                    current_neuron.rgb_tuple,
                    new_x,
                    new_y
                )
                self.extended_map[i][j] = new_neuron

    def compute_maps(self):
        """
        self.full_map a map containing som neurons
        :return:
        """
        self.full_map = [[0 for _ in range(self.result_loader.som_matrix_width + 1)] for _ in
                         range(self.result_loader.som_matrix_height + 1)]
        for entry in self.som_neurons:
            self.full_map[entry.matrix_x_coord][entry.matrix_y_coord] = entry
        for i in range(len(self.som_chromosomes)):
            chromosome = self.som_chromosomes[i]
            for neuron in self.som_neurons:
                if chromosome.x == neuron.x and chromosome.y == neuron.y:
                    self.som_chromosomes[i].matrix_x_coord = neuron.matrix_x_coord
                    self.som_chromosomes[i].matrix_y_coord = neuron.matrix_y_coord
                    break
            else:
                print("[ERROR] Can't find matching neuron for chromosome: %s" % chromosome)
        # self.extended_map = [[self.full_map for _ in range(3)] for _ in range(3)]
        self.__compute_extended_map()

    def __get_lee_distances_for_neuron(self, ch1):
        """
        distanta vafi calculata pe extended map

        :param ch1: SOMChromosomeResultEntry Object (ch_id, color_RGB, ch_img_path, x, y)

        SOMNeuronResultEntry(neuron_id, matrix_x_coord, matrix_y_coord, color_RGB, x, y, value)
        :return: distanta minima dintre ch1 si ch2, tinand cont de culorile de pe harta ( folosind lee)
        """
        dist_matrix = [
            [-1 for _ in range(len(self.extended_map[0]))]
            for _ in range(len(self.extended_map))
        ]
        dist_matrix[ch1.matrix_x_coord][ch1.matrix_y_coord] = self.extended_map[ch1.matrix_x_coord][
            ch1.matrix_y_coord].value
        points = [(ch1.matrix_x_coord, ch1.matrix_y_coord)]
        min_i = self.extended_map[0][0].matrix_x_coord
        min_j = self.extended_map[0][0].matrix_y_coord
        max_i = self.extended_map[len(self.extended_map) - 1][0].matrix_x_coord
        max_j = self.extended_map[0][len(self.extended_map[0]) - 1].matrix_y_coord
        while len(points) > 0:
            point = points[0]
            points = points[1:]
            i = point[0]
            j = point[1]
            # n_i - neighbour_i
            # n_j - neighbour_j
            for (n_i_coord, n_j_coord) in [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]:
                n_i = i + n_i_coord
                n_j = j + n_j_coord
                if min_i <= n_i <= max_i and min_j <= n_j <= max_j:
                    if (dist_matrix[n_i][n_j] == -1) or \
                            dist_matrix[n_i][n_j] \
                            > dist_matrix[i][j] + self.extended_map[n_i][n_j].value:
                        points.append((n_i, n_j))
                        dist_matrix[n_i][n_j] = \
                            dist_matrix[i][j] + self.extended_map[n_i][n_j].value

        return dist_matrix

    def __get_distance_between_chromosomes(self, ch1, ch2):
        map_h = len(self.full_map)
        map_w = len(self.full_map[0])
        ch1_neuron = self.extended_map[ch1.matrix_x_coord][ch1.matrix_y_coord]
        dist_matrix = self.__get_lee_distances_for_neuron(ch1_neuron)
        center = dist_matrix[ch2.matrix_x_coord][ch2.matrix_y_coord]
        nv = dist_matrix[ch2.matrix_x_coord - map_h][ch2.matrix_y_coord - map_w]
        n = dist_matrix[ch2.matrix_x_coord - map_h][ch2.matrix_y_coord]
        ne = dist_matrix[ch2.matrix_x_coord - map_h][ch2.matrix_y_coord + map_w]
        e = dist_matrix[ch2.matrix_x_coord][ch2.matrix_y_coord + map_w]
        se = dist_matrix[ch2.matrix_x_coord + map_h][ch2.matrix_y_coord + map_w]
        s = dist_matrix[ch2.matrix_x_coord + map_h][ch2.matrix_y_coord]
        sv = dist_matrix[ch2.matrix_x_coord + map_h][ch2.matrix_y_coord - map_w]
        v = dist_matrix[ch2.matrix_x_coord][ch2.matrix_y_coord - map_w]
        return min(center, nv, n, ne, e, se, s, sv, v)

    def get_dist_matrix(self, reeval=True):
        if self.ch_dist_matrix != list():
            return self.ch_dist_matrix
        if os.path.isfile(self.__dist_matrix_file) and not reeval:
            self.__load_dist_matrix_from_file()
        if self.ch_dist_matrix == list():
            self.__compute_distance_matrix()
        return self.ch_dist_matrix

    def __compute_distance_matrix_aux(self, i, entry_i):
        for j in range(i + 1, len(self.som_chromosomes)):
            entry_j = self.som_chromosomes[j]
            # if i == j:
            #     current_ch_dist_list.append(self.extended_map[entry_j.matrix_x_coord][entry_j.matrix_y_coord].value)
            # else:
            print(str(i) + " " + str(j))
            current_dist = self.__get_distance_between_chromosomes(entry_i, entry_j)
            self.ch_dist_matrix[i][j] = current_dist
            self.ch_dist_matrix[j][i] = current_dist
            # self.ch_dist_matrix[i] = current_ch_dist_list

    def __compute_distance_matrix(self):
        if self.full_map == list():
            self.compute_maps()
        from threading import Thread
        self.ch_dist_matrix = [[0 for _ in range(len(self.som_chromosomes))] for _ in range(len(self.som_chromosomes))]
        th_list = list()
        for i in range(len(self.som_chromosomes)):
            entry_i = self.som_chromosomes[i]
            # current_ch_dist_list = [0 for _ in range(i+1)]
            t = Thread(target=self.__compute_distance_matrix_aux, args=[i, entry_i])
            t.start()
            th_list.append(t)
            print("Add thread %d" % i)
            # self.__compute_distance_matrix_aux(i, entry_i)
        i = len(th_list) - 1
        for th in th_list[::-1]:
            print("join thread %d" % i)
            i -= 1
            if th is not None:
                th.join()
        self.__dump_ch_dist_matrix_in_file()

    def __compute_distance_matrix_old(self):
        if self.full_map == list():
            self.compute_maps()
        self.ch_dist_matrix = [[0 for _ in range(len(self.som_chromosomes))] for _ in range(len(self.som_chromosomes))]
        for i in range(len(self.som_chromosomes)):
            entry_i = self.som_chromosomes[i]
            # current_ch_dist_list = [0 for _ in range(i+1)]
            for j in range(i + 1, len(self.som_chromosomes)):
                entry_j = self.som_chromosomes[j]
                # if i == j:
                #     current_ch_dist_list.append(self.extended_map[entry_j.matrix_x_coord][entry_j.matrix_y_coord].value)
                # else:
                print(str(i) + " " + str(j))
                current_dist = self.__get_distance_between_chromosomes(entry_i, entry_j)
                self.ch_dist_matrix[i][j] = current_dist
                self.ch_dist_matrix[j][i] = current_dist
                # self.ch_dist_matrix[i] = current_ch_dist_list
        self.__dump_ch_dist_matrix_in_file()

    def get_karyotype(self):
        if self.karyotype == list():
            self.__compute_karyotype()
        return self.karyotype

    def __compute_karyotype(self):
        if self.ch_dist_matrix == list():
            self.get_dist_matrix()
        chosen = list()
        old_len_chosen = len(chosen) - 2
        while old_len_chosen != len(chosen):
            min_dist = 9999999999
            old_len_chosen = len(chosen)
            print("Chosen: %s" % str(chosen))
            min_i = -1
            min_j = -1
            for i in range(len(self.ch_dist_matrix)):
                if i not in chosen:
                    for j in range(len(self.ch_dist_matrix[0])):
                        if min_dist > self.ch_dist_matrix[i][j] > 0 and j not in chosen and i != j:
                            min_i = i
                            min_j = j
                            min_dist = self.ch_dist_matrix[i][j]
            if min_i != -1 and min_j != -1:
                self.karyotype.append([self.som_chromosomes[min_i], self.som_chromosomes[min_j]])
                chosen.append(min_i)
                chosen.append(min_j)
                print("Min dist = %s" % str(min_dist))
                print("Min i = %d" % min_j)
                print("Min j = %d" % min_j)
                print(self.som_chromosomes[min_i])
                print(self.som_chromosomes[min_j])
            print("===================================================================================================")
        if len(self.ch_dist_matrix) != 2 * len(self.karyotype):
            for i in range(len(self.ch_dist_matrix)):
                if i not in chosen:
                    self.karyotype.append([self.som_chromosomes[i]])

        print("len chosen: %d" % len(self.karyotype))
        print("len som results: %d" % len(self.som_chromosomes))
        # print(sorted(self.karyotype))

    def __load_dist_matrix_from_file(self):
        """
            File format:
            n,m
            n lines and c columns
            :return:
        """
        try:
            with open(self.__dist_matrix_file, "r") as f:
                content = f.read()
                lines = content.split("\n")
                first_line = lines[0]
                lines = lines[1:]
                n = int(first_line.split(",")[0].strip())
                m = int(first_line.split(",")[1].strip())
                self.ch_dist_matrix = list()
                for i in range(0, n):
                    current_dist_matrix_line = list()
                    for elem in lines[i].split(CH_DIST_MATRIX_FILE_VALUES_SEPARATOR):
                        dist = float(elem.strip())
                        current_dist_matrix_line.append(dist)
                    self.ch_dist_matrix.append(current_dist_matrix_line)
        except:
            print("Exception occurred while trying to load dist matrix from file. Traceback: %s"
                  % traceback.format_exc())
            self.ch_dist_matrix = list()

    def __dump_ch_dist_matrix_in_file(self):
        """
        File format:
        n,m
        n lines and c columns
        :return:
        """
        with open(self.__dist_matrix_file, "w") as g:
            g.write(str(len(self.ch_dist_matrix)))
            g.write(",")
            g.write(str(len(self.ch_dist_matrix[0])))
            g.write("\n")
            for dist_line in self.ch_dist_matrix:
                line_to_write = ""
                for entry in dist_line:
                    line_to_write += str(entry) + ","
                g.write(line_to_write[:-1] + "\n")

    def get_dist_matrix_file_path(self):
        if len(self.ch_dist_matrix) < 0:
            self.get_dist_matrix()

        return self.__dist_matrix_file


def interpret_som_result(
        som_results_file_path=r'D:\GIT\Karyotyping-Project\PythonProject\D_PostProcessSOMResults\13_mai_all_features_37.txt.out',
        karyo_pairs_file=r'D:\GIT\Karyotyping-Project\PythonProject\Z_Images\kar-segm\1_test_1apr\pairs.txt'
):
    karyo_pairs_file_out = som_results_file_path[:-4] + "_pairs.out"
    obj = SOMResultsV2(som_results_file_path)
    # obj.pretty_print_som_results()
    a = obj.get_dist_matrix()
    s = [[str(e) for e in row] for row in obj.ch_dist_matrix]
    lens = [max(map(len, col)) for col in zip(*s)]
    fmt = '\t'.join('{{:{}}}'.format(x) for x in lens)
    table = [fmt.format(*row) for row in s]
    print('\n'.join(table))
    import D_PostProcessSOMResults.karytype_image_generator as karytype_image_generator
    k_obj = karytype_image_generator.KaryotypeImageGenerator(obj.get_karyotype(), som_results_file_path)
    try:
        k_obj.generate_karyotype_image()
    except:
        print("Failed to generate karyotype image. Traceback: %s" % traceback.format_exc())
    if os.path.isfile(karyo_pairs_file):
        expected_karyotype = ExpectedKaryotype(karyo_pairs_file).load()
        with open(karyo_pairs_file_out, 'w') as g:
            for karyotype_entry in expected_karyotype:
                if len(karyotype_entry) > 1:
                    for i in range(len(karyotype_entry)):
                        for j in range(i + 1, len(karyotype_entry)):
                            g.write("Dist[%d][%d] = %s\n" % (i, j, obj.ch_dist_matrix[i][j]))
                else:
                    g.write("Dist[%d][-] = -\n" % karyotype_entry[0])
    return obj.get_dist_matrix_file_path(), k_obj.karyotype_image_path


if __name__ == "__main__":
    interpret_som_result()
