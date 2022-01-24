import os

import pygmo
from pygmo import hypervolume
import numpy as np

M = 5  # objectives
C = 4  # Constraints
d = 32  # input dimensions
seed = 1
reference_point = [1.0, 1.0, 1.0, 1.0, 1.0]
paths = '.'

algs = ["MOEAD", "NSGAII", "MESMOC"]

for alg in algs:

    filename = "../plotter/results/" + alg + ".txt"
    f = open(filename, "r")
    f1 = f.readlines()
    inits = [x.split() for x in f1]
    initial_number = len(inits)
    f.close()
    valid_points = []

    for k in range(initial_number):
        y_new = [(float(p)) for p in inits[k]]
        functions = y_new[:M]
        # print(functions)
        constraints = y_new[M:]
        functions[0] = (functions[0]+45.0)/50.0
        functions[1] = (functions[1]+0.2)/2.01
        functions[2] /= 100.0
        functions[3] = (functions[3]+0.0181)/0.0181
        functions[4] = (functions[4]+0.16)/0.121

        if constraints[0]>0 and min(constraints[1:])>=0:
            valid_points.append(tuple([-functions[t] for t in range(M)]))

            with open(os.path.join(paths, '../plotter/hvs/hypervolumes_' + alg + '.txt'), "a") as filehandle:
                if len(valid_points)>1:
                    hv = hypervolume((np.asarray(valid_points))).compute(reference_point)
                    filehandle.write(str(hv) + "\n")
                else:
                    filehandle.write("0" + "\n" )
            filehandle.close()

    ndf, dl, dc, ndr = pygmo.fast_non_dominated_sorting(valid_points)
    for i in ndf[0]:
        print(valid_points[i])

    print("****************************************")
