import os

import pygmo
from pygmo import hypervolume
import numpy as np

M = 2  # objectives
C = 6  # Constraints
d = 32  # input dimensions
seed = 1
reference_point = [1.0, 1.0]
paths = '.'

algs = ["NSGAII", "MOEAD", "MESMOC"]

for alg in algs:

    filename = "../plotter/results/" + alg + ".txt"

    f = open(filename, "r")
    f1 = f.readlines()
    inits = [x.split() for x in f1]
    initial_number = len(inits)
    f.close()
    valid_points = []

    hvs = []

    for k in range(initial_number):
        y_new = [(float(p)) for p in inits[k]]
        functions = y_new[:M]
        # print(functions)
        constraints = y_new[M:]

        if constraints[0] > 0 and min(constraints[2:]) >= 0:
            valid_points.append(tuple([-functions[t] for t in range(M)]))
            if len(valid_points) > 1:
                hv = hypervolume((np.asarray(valid_points))).compute(reference_point)
                hvs.append(hv)

    print(len(valid_points), " are valid out of ", initial_number, " with 32 initial valid points")

    valid_points = valid_points[32:]

    # l = max(hvs)
    # t = min(hvs)
    # for i in range(len(hvs)):
    #     hvs[i] = (hvs[i]-t)/(l-t)

    with open(os.path.join(paths, '../plotter/hvs/hypervolumes_' + alg + '.txt'), "a") as filehandle:
        for h in hvs:
            filehandle.write(str(h) + "\n")
    filehandle.close()

    ndf, dl, dc, ndr = pygmo.fast_non_dominated_sorting(valid_points)
    print(ndf)
    for i in ndf[0]:
        print([-j for j in valid_points[i]])

    print("****************************************")
