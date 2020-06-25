from matplotlib import pyplot as plt
from matplotlib.axes import Axes


class Graph2D:

    def __init__(self, aqe_file):
        self._aqe_data_file = aqe_file
        self.x_list = list()
        self.y_list = list()
        self.x_label = "Iteration"
        self.y_label = "AQE value"
        self.__load_data_from_file()

    def generate(self):
        plt.xlabel(self.x_label)
        plt.title("Average Quantization of Error")
        plt.ylabel(self.y_label)
        plt.plot(self.x_list, self.y_list, '-')
        plt.savefig("aqe.png")
        print("aici")
        # plt.hold()

    def __load_data_from_file(self):
        with open(self._aqe_data_file, "r") as f:
            for line in f.readlines():
                line = line.strip()
                if line != "":
                    epoch_s, value_s = line.split(",")[0], line.split(",")[1]
                    epoch = int(epoch_s.strip())
                    value = float(value_s.strip())
                    self.x_list.append(epoch)
                    self.y_list.append(value)


if __name__ == "__main__":
    obj = Graph2D(r'D:\GIT\Karyotyping-Project\PythonProject\D_PostProcessSOMResults\good_aqw_1.txt')
    obj.generate()