import math

import D_PostProcessSOMResults.SOM_result_loader as SOM_result_loader
import D_PostProcessSOMResults.SOM_result_entry as SOM_result_entry


class SOMResultsV1:
    def __init__(self, results_file_path):
        self.results_file_path = results_file_path
        self.som_chromosomes, self.som_neurons = SOM_result_loader.SOMResultsLoader(results_file_path).load_results()
        self.extended_map = list()
        self.full_map = list()
        self.ch_dist_matrix = list()
        self.karyotype = list()

    def pretty_print_som_results(self):
        for entry in self.som_chromosomes:
            print("ID: %s" % entry.ch_id)
            print("\tColor code: [R:%s G:%s B:%s]" % (entry.rgb_tuple[0], entry.rgb_tuple[1], entry.rgb_tuple[2]))
            print("\tImage path: %s" % entry.ch_img_path)
            print("\tSOM Torus Coordinates: [X:%s Y:%s]" % (entry.x, entry.y))

    def compute_maps(self):
        max_x, max_y = self.__get_map_dimensions()
        self.full_map = [[0 for _ in range(max_y + 1)] for _ in range(max_x + 1)]
        for entry in self.som_chromosomes:
            self.full_map[entry.x][entry.y] = entry.ch_id
        self.extended_map = [[self.full_map for _ in range(3)] for _ in range(3)]

    @staticmethod
    def __get_euclidean_distance_between_points(ch1, ch2):
        """

        :param ch1: tuple or list (x1, y1), [x1, y1]
        :param ch2: tuple or list (x2, y2), [x2, y2]
        :return: euclidean distance between ch1 and ch2
        """
        return math.sqrt((ch2[0] - ch1[0]) ** 2 + (ch2[1] - ch1[1]) ** 2)

    def __get_distance_between_chromosomes(self, ch1, ch2):
        map_h = len(self.full_map)
        map_w = len(self.full_map[0])
        center = self.__get_euclidean_distance_between_points([ch1.x, ch1.y], [ch2.x, ch2.y])
        nv = self.__get_euclidean_distance_between_points([ch1.x, ch1.y], [ch2.x-map_h, ch2.y-map_w])
        n = self.__get_euclidean_distance_between_points([ch1.x, ch1.y], [ch2.x-map_h, ch2.y])
        ne = self.__get_euclidean_distance_between_points([ch1.x, ch1.y], [ch2.x-map_h, ch2.y+map_w])
        e = self.__get_euclidean_distance_between_points([ch1.x, ch1.y], [ch2.x, ch2.y+map_w])
        se = self.__get_euclidean_distance_between_points([ch1.x, ch1.y], [ch2.x+map_h, ch2.y+map_w])
        s = self.__get_euclidean_distance_between_points([ch1.x, ch1.y], [ch2.x+map_h, ch2.y])
        sv = self.__get_euclidean_distance_between_points([ch1.x, ch1.y], [ch2.x+map_h, ch2.y-map_w])
        v = self.__get_euclidean_distance_between_points([ch1.x, ch1.y], [ch2.x, ch2.y-map_w])
        return min(center, nv, n, ne, e, se, s, sv, v)

    def get_dist_matrix(self):
        if self.ch_dist_matrix == list():
            self.__compute_distance_matrix()
        return self.ch_dist_matrix

    def __compute_distance_matrix(self):
        if self.full_map == list():
            self.compute_maps()
        for i in range(len(self.som_chromosomes)):
            entry_i = self.som_chromosomes[i]
            current_ch_dist_list = list(i*[0])
            for j in range(i, len(self.som_chromosomes)):
                entry_j = self.som_chromosomes[j]
                current_ch_dist_list.append(self.__get_distance_between_chromosomes(entry_i, entry_j))
            self.ch_dist_matrix.append(current_ch_dist_list)

    def __get_map_dimensions(self):
        max_x = 0
        max_y = 0
        for entry in self.som_chromosomes:
            if entry.x > max_x:
                max_x = entry.x
            if entry.y > max_y:
                max_y = entry.y
        return max_x, max_y

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
            for i in range(len(a)):
                if i not in chosen:
                    for j in range(len(a[0])):
                        if min_dist > self.ch_dist_matrix[i][j] > 0 and j not in chosen:
                            min_i = i
                            min_j = j
                            min_dist = a[i][j]
            if min_i != -1 and min_i != -1:
                self.karyotype.append([self.som_chromosomes[min_i], self.som_chromosomes[min_j]])
                chosen.append(min_i)
                chosen.append(min_j)
                print("Min dist = %s" % str(min_dist))
                print("Min i = %d" % min_j)
                print("Min j = %d" % min_j)
                print(self.som_chromosomes[min_i])
                print(self.som_chromosomes[min_j])
            print("===================================================================================================")
        print("len chosen: %d" % len(self.karyotype))
        print("len som results: %d" % len(self.som_chromosomes))
        # print(sorted(self.karyotype))


if __name__ == "__main__":
    # som_results_file_path = r'D:\GIT\Karyotyping-Project\PythonProject\D_PostProcessSOMResults\to_extract_features_test.txt.out'
    som_results_file_path = r'D:\GIT\Karyotyping-Project\PythonProject\D_PostProcessSOMResults\to_extract_features_test_2x_len.txt'
    obj = SOMResultsV1(som_results_file_path)
    obj.pretty_print_som_results()
    a = obj.get_dist_matrix()
    s = [[str(e) for e in row] for row in a]
    lens = [max(map(len, col)) for col in zip(*s)]
    fmt = '\t'.join('{{:{}}}'.format(x) for x in lens)
    table = [fmt.format(*row) for row in s]
    print('\n'.join(table))
    obj.get_karyotype()
    import D_PostProcessSOMResults.karytype_image_generator as karytype_image_generator
    k_obj = karytype_image_generator.KaryotypeImageGenerator(obj.get_karyotype(), som_results_file_path)
    k_obj.generate_karyotype_image()
