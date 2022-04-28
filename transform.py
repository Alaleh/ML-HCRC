import copy
import numpy as np


def convert_to_original(x1):
    x = copy.deepcopy(x1)
    for i in range(12):
        if x[i] > 0.66:
            x[i] = 25.0
        elif x[i] > 0.33:
            x[i] = 20.0
        else:
            x[i] = 15.0
    x[12:24] = np.round(list(40 + np.asarray(x[12:24]) * (200 - 40)))
    C_values = [4.5, 4.7, 5, 5.6, 6.6, 6.8, 7.5, 8.2, 9.4, 10.0, 12.0, 14.0, 15.0, 17.0, 18.0, 19.0, 20.0, 22.0]
    for i in range(24, 28):
        t = x[i] * (22.0 - 4.5) + 4.5
        for j in range(1, len(C_values)):
            if C_values[j - 1] <= t < C_values[j]:
                if t - C_values[j - 1] > C_values[j] - t:
                    x[i] = C_values[j]
                else:
                    x[i] = C_values[j - 1]
    l_values = [.220, .240, .250, .270, .300, .330, .360, .390, .400, .420, .470, .500, .530, .560, .590, .600, .620,
                .680, .700, .760, .770, .820, .900, 1, 1.2, 1.5, 1.8, 2.2, 3.3, 4.7]
    t = x[28] * (4.7 - .220) + .220
    for j in range(1, len(l_values)):
        if l_values[j - 1] <= t < l_values[j]:
            if t - l_values[j - 1] > l_values[j] - t:
                x[28] = l_values[j]
            else:
                x[28] = l_values[j - 1]
    x[29] = np.round(500 + np.asarray(x[29]) * (4000 - 500)) // 100.0 * 100.0
    x[30] = np.round(200 + np.asarray(x[30]) * (3000 - 200)) // 20.0 * 20.0
    x[31] = np.round(550 + np.asarray(x[31]) * (630 - 550)) // 5.0 * 5.0
    return x


def convert_to_01(x1):
    x = copy.deepcopy(x1)

    for i in range(12):
        if x[i] == 25.0:
            x[i] = 0.8
        elif x[i] == 20.0:
            x[i] = 0.5
        else:
            x[i] = 0.1

    for i in range(12, 24):
        x[i] = (x[i] - 40) / (200 - 40)

        # C_1 to C_4 : one of the following values:
    for i in range(24, 28):
        x[i] = (x[i] - 4.5) / (22.0 - 4.5)

     # l
    x[28] = (x[28] - 0.220) / (4.7 - .220)
    # cpo
    x[29] = (x[29] - 200) / (6000 - 200)
    # ramp
    x[30] = (x[30] - 200) / (3000 - 200)
    # vref
    x[31] = (x[31] - 550) / (630 - 550)

    return x
