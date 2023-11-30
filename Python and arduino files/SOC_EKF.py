import numpy as np
import serial
from SOC_OCV_curve_fit import *
from matplotlib import pyplot as plt

arduino = serial.Serial(port='COM9', baudrate=115200, timeout=0.1)

SOC_initial = 1
X = np.array([[SOC_initial], [0]])

Qn = 3 * 3600
delta_t = 5.0

SOC_OCV = best_fit()
dSOC_OCV = np.polyder(SOC_OCV)

Q = np.array([[1e-5, 0], [0, 1e-4]])
R = np.array([0.004])
P = np.array([[0.025, 0], [0, 0.01]])

R0 = 1.611e-4
R1 = 0.116
C1 = 10.386

# R0 = 1.5581e-5
# R1 = 0.0031
# C1 = 1.8565e4

tau_1 = R1 * C1
a1 = np.exp(-delta_t / tau_1)
b1 = R1 * (1 - np.exp(-delta_t / tau_1))

A = np.array([[1, 0], [0, a1]])
B = np.array([[-1 * 0.9 / (Qn * 3600)], [b1]])
D = np.array([-R0])

fig = plt.figure(figsize=(12, 10))
plot = fig.add_subplot(111)

# plt.xlim([-5, plot_width + 5])
plt.ylim([0, 100])

prev_SOC_estimate = SOC_initial


def read_data(ser):
    line = ser.readline()
    line = line.decode("utf-8")
    data = line.strip()
    return data


t = 0

while True:
    raw_data = read_data(arduino)
    if raw_data:
        t += 5
        U = float(raw_data.split("$")[0])
        V = float(raw_data.split("$")[1])
        SOC = X[0][0]
        V1 = X[1][0]
        OCV = np.polyval(SOC_OCV, SOC)
        Vt = OCV - V1 - (U * R0)
        dOCV = np.polyval(dSOC_OCV, SOC)
        C = np.array([dOCV, -1]).reshape([1, 2])
        error = V - Vt
        SOC_hat = X[0][0]
        X = np.dot(A, X) + np.dot(B, U)
        P = (A @ P @ A.T) + Q
        KalmanGain = (P @ C.T) / ((C @ P @ C.T) + R)
        X = X + (KalmanGain * error)
        P = (np.ones(shape=[2, 2]) - (KalmanGain @ C)) @ P

        plot.scatter([t], [SOC_hat * 100], color="blue", s=50)
        print(f"t : {t}    SOC : {SOC_hat}   Current : {U}   Voltage : {V}      Capacitor Voltage : {V1}")
        plt.pause(0.001)


plt.show()
