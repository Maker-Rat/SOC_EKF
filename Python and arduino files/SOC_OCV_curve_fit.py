import numpy as np
import matplotlib.pyplot as plt
import csv


def best_fit():
    with open('Samsung_18650_Raw_OCV_Data.csv', mode='r') as file:
        data_file = csv.reader(file)
        data = np.array(list(data_file))

    OCV = np.float_(data[1:, 1])
    SOC = np.float_(data[1:, 2])
    SOC = SOC/100.0

    poly = np.polyfit(SOC, OCV, deg=10)

    # fig, ax = plt.subplots()
    # ax.plot(SOC, OCV, label='data')
    # ax.plot(SOC, np.polyval(poly, SOC), label='fit')
    # ax.legend()
    # plt.show()

    return poly
