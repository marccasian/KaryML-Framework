from D_PostProcessSOMResults.SOM_results_constants import *
import D_PostProcessSOMResults.SOM_result_chromosome_entry as SOM_result_chromosome_entry
import D_PostProcessSOMResults.SOM_result_neuron_entry as SOM_result_neuron_entry


class SOMResultsLoader:
    def __init__(self, results_file_path):
        self.__results_file_path = results_file_path
        self.number_of_chromosomes = 0
        self.number_of_neurons = 0
        self.som_width = 0
        self.som_height = 0
        self.som_matrix_width = 0
        self.som_matrix_height = 0
        self.result_chromosomes = list()
        self.result_neurons = list()

    def load_results(self):
        if len(self.result_chromosomes) > 0 and len(self.result_neurons) > 0:
            return self.result_chromosomes, self.result_neurons
        self.result_chromosomes = list()
        with open(self.__results_file_path, "r") as f:
            lines = f.readlines()
            self.number_of_chromosomes = int(lines[0].strip())
            chromosomes_lines = lines[1:self.number_of_chromosomes + 1]
            for result_chromosome_line in chromosomes_lines:
                result_chromosome_line = result_chromosome_line.strip()
                if result_chromosome_line.strip() != "":
                    result_chromosome_entry = self.__get_result_chromosome_from_result_line(result_chromosome_line)
                    self.result_chromosomes.append(result_chromosome_entry)
            self.som_matrix_height = int(lines[self.number_of_chromosomes + 1].strip().split(RESULT_ENTRY_SEPARATOR)[0])
            self.som_matrix_width = int(lines[self.number_of_chromosomes + 1].strip().split(RESULT_ENTRY_SEPARATOR)[1])
            self.number_of_neurons = self.som_matrix_height * self.som_matrix_width
            self.som_width, self.som_height = \
                (int(lines[self.number_of_chromosomes + 2].strip().split(RESULT_ENTRY_SEPARATOR)[0].strip()),
                 int(lines[self.number_of_chromosomes + 2].strip().split(RESULT_ENTRY_SEPARATOR)[1].strip()))
            neurons_lines = lines[self.number_of_chromosomes + 3:]
            for result_neuron_line in neurons_lines:
                result_neuron_line = result_neuron_line.strip()
                if result_neuron_line.strip() != "":
                    result_neuron_entry = self.__get_result_neuron_from_result_line(result_neuron_line)
                    self.result_neurons.append(result_neuron_entry)

        return self.result_chromosomes, self.result_neurons

    @staticmethod
    def __get_result_chromosome_from_result_line(result_chromosome_line):
        result_chromosome_line_parts = result_chromosome_line.split(RESULT_ENTRY_SEPARATOR)
        color_parts = result_chromosome_line_parts[CHROMOSOME_RESULT_ENTRY_RGB_COMPONENT_INDEX].split(
            COLOR_COMPONENT_SEPARATOR)
        color = tuple(color_parts)
        return SOM_result_chromosome_entry.SOMChromosomeResultEntry(
            result_chromosome_line_parts[CHROMOSOME_RESULT_ENTRY_ID_INDEX],
            color,
            result_chromosome_line_parts[CHROMOSOME_RESULT_ENTRY_IMAGE_PATH_INDEX],
            int(float(result_chromosome_line_parts[CHROMOSOME_RESULT_ENTRY_SOM_TORUS_X_COORD_INDEX])),
            int(float(result_chromosome_line_parts[CHROMOSOME_RESULT_ENTRY_SOM_TORUS_Y_COORD_INDEX]))
        )

    @staticmethod
    def __get_result_neuron_from_result_line(result_neuron_line):
        result_neuron_line_parts = result_neuron_line.split(RESULT_ENTRY_SEPARATOR)
        color_parts = (int(i) for i in result_neuron_line_parts[NEURON_RESULT_ENTRY_RGB_COMPONENT_INDEX].split(
            COLOR_COMPONENT_SEPARATOR))
        color = tuple(color_parts)
        return SOM_result_neuron_entry.SOMNeuronResultEntry(
            result_neuron_line_parts[NEURON_RESULT_ENTRY_ID_INDEX],
            int(float(result_neuron_line_parts[NEURON_RESULT_ENTRY_MATRIX_X_COORD_INDEX])),
            int(float(result_neuron_line_parts[NEURON_RESULT_ENTRY_MATRIX_Y_COORD_INDEX])),
            color,
            int(float(result_neuron_line_parts[NEURON_RESULT_ENTRY_SOM_TORUS_X_COORD_INDEX])),
            int(float(result_neuron_line_parts[NEURON_RESULT_ENTRY_SOM_TORUS_Y_COORD_INDEX]))
        )
