class SOMNeuronResultEntry:
    def __init__(self, neuron_id, matrix_x_coord, matrix_y_coord, color_RGB, x, y):
        """
        4;0.0;4.0;3:3:3;46.86274509803921;114.66000000000001
        """
        self.neuron_id = neuron_id
        self.rgb_tuple = color_RGB
        self.matrix_x_coord = matrix_x_coord
        self.matrix_y_coord = matrix_y_coord
        self.x = x
        self.y = y
        self.value = ((sum(self.rgb_tuple))/3)/255

    def __str__(self):
        return "Neuron ID: %s\n\tRGB: %s\n\tMatrix X= %s\n\tMatrix Y = %s\n\tX = %s\n\tY = %s" \
               % (str(self.neuron_id), str(self.rgb_tuple),
                  str(self.matrix_x_coord), str(self.matrix_y_coord),
                  str(self.x), str(self.y))

    def __eq__(self, other):
        return self.neuron_id == other.ch_id \
               and self.rgb_tuple == other.rgb_tuple \
               and self.matrix_x_coord == other.matrix_x_coord \
               and self.matrix_y_coord == other.matrix_y_coord
