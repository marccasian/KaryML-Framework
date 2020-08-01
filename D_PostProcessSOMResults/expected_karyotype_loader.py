import traceback
from typing import List

from a_Common.my_logger import get_new_logger


class ExpectedKaryotype:
    def __init__(self, expected_karyotype_file):
        self.expected_karyotype_file = expected_karyotype_file
        self.karyotype: List[List[int, int]] = list()
        self.logger = get_new_logger(self.__class__.__name__)

    def load(self):
        """
            File format:
            id_ch_1, id_ch_2
            ...
            id_ch_n
            :return: List[List[ch1,<ch2>,<ch3>], ...]
        """
        try:
            with open(self.expected_karyotype_file, "r") as f:
                for line in f.readlines():
                    line = line.strip()
                    if line != "":
                        c1 = line
                        c2 = None
                        c3 = None
                        if ',' in line:
                            components = line.split(",")
                            if len(components) == 2:
                                c1, c2 = line.split(",")
                            elif len(components) == 3:
                                c1, c2, c3 = line.split(",")
                        c1 = int(c1.strip())
                        if c2 is not None:
                            c2 = int(c2.strip())
                            if c3 is None:
                                self.karyotype.append(sorted([c1, c2]))
                            else:
                                c3 = int(c3.strip())
                                self.karyotype.append(sorted([c1, c2, c3]))
                        else:
                            self.karyotype.append([c1])
        except:
            self.logger.exception("Exception occurred while trying to load pairs from file. Traceback: %s"
                                  % traceback.format_exc())
        return self.karyotype

    def get_pair_id(self, chromosome_id: int):
        if len(self.karyotype) == 0:
            self.load()
        pair_id = 1
        for pair in self.karyotype:
            if chromosome_id in pair:
                return pair_id
            pair_id += 1
        else:
            raise ValueError("Did not found ch {} in karyotype {}".format(chromosome_id, self))

    def __str__(self):
        return ";".join(["-".join([str(j) for j in i]) for i in self.karyotype])


if __name__ == '__main__':
    k = ExpectedKaryotype(r"..\__disertation_experiments\dataset\6\6_expected_karyotype.txt")
    k.load()
    k.get_pair_id(1)
