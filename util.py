import re
import os


def csv_read_bac(path):
    with open(path) as file:
        den = np.loadtxt(file, delimiter = ",")
        den_quarter = np.zeros((int(den.shape[0] / 4), int(den.shape[1] / 4)))
        # print(den_quarter.shape)
        for i in range(len(den_quarter)):
            for j in range(len(den_quarter[0])):
                for p in range(4):
                    for q in range(4):
                        den_quarter[i][j] += den[i * 4 + p][j * 4 + q]

    print(path,"     LOADED")
    return den_quarter

