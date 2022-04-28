import os

import pygmo
from pygmo import hypervolume
import numpy as np

M = 5  # objectives
C = 4  # Constraints
d = 32  # input dimensions
seed = 1
reference_point = [0.01, 1.01, 1.01, 1.01, 1000.01]
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

        if alg == "MESMOC":
            functions = y_new[:M]
            constraints = [y_new[M]]+y_new[M+2:]
        else:
            functions = [y_new[2], y_new[3], y_new[4], y_new[1], y_new[0]]
            constraints = y_new[M:]

        if constraints[0] > 0 and min(constraints[1:]) >= 0:
            valid_points.append(tuple([-functions[t] for t in range(M)]))
            if len(valid_points) > 1:
                hv = hypervolume((np.asarray(valid_points))).compute(reference_point)
                hvs.append(hv)

    print(len(valid_points), " are valid out of ", initial_number, " with 32 initial valid points")

    valid_points = valid_points[32:]

    with open(os.path.join(paths, '../plotter/hvs/hypervolumes_' + alg + '.txt'), "a") as filehandle:
        for h in hvs:
            filehandle.write(str(h) + "\n")
    filehandle.close()

    ndf, dl, dc, ndr = pygmo.fast_non_dominated_sorting(valid_points)
    print(ndf)
    for i in ndf[0]:
        print([-j for j in valid_points[i]])

    print("****************************************")
