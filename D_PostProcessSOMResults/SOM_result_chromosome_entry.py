class SOMChromosomeResultEntry:
    def __init__(self, ch_id, color_RGB, ch_img_path, x, y):
        """
        0;65:65:65;D:\GIT\Karyotyping-Project\PythonProject\Z_Images\kar-segm\1_test_1apr\to_extract\0.bmp;256.0;220.0
        """
        self.ch_id = ch_id
        self.rgb_tuple = color_RGB
        self.ch_img_path = ch_img_path
        self.x = x
        self.y = y
        self.matrix_x_coord = -1
        self.matrix_y_coord = -1
        self.image_obj = None

    def __str__(self):
        return "Ch ID: %s\n\tRGB: %s\n\tImg path: %s\n\tX = %s\n\tY = %s" % (
            str(self.ch_id), str(self.rgb_tuple), str(self.ch_img_path), str(self.x), str(self.y))

    def __eq__(self, other):
        return self.ch_id == other.ch_id \
               and self.rgb_tuple == other.rgb_tuple \
               and self.ch_img_path == other.ch_img_path
