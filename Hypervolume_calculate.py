import os
from pygmo import hypervolume
import numpy as np

M = 4  # objectives
C = 5  # Constraints
d = 32  # input dimensions
seed = 1
reference_point = [1, 1, 1, 1]
paths = '.'

alg = "NSGAII"

filename = "results/" + alg + ".txt"
f = open(filename, "r")
f1 = f.readlines()
inits = [x.split() for x in f1]
initial_number = len(inits)
f.close()
valid_points = []

for k in range(initial_number):
    y_new = [(float(p)) for p in inits[k]]
    functions = y_new[:M]
    constraints = y_new[M:]
    functions[1] /= 100.0

    if constraints[0]>0 and min(constraints[1:])>=0:
        valid_points.append(tuple([-functions[t] for t in range(M)]))
        print(k, constraints)

        with open(os.path.join(paths, 'hpvs/hypervolumes_' + alg + '.txt'), "a") as filehandle:
            if len(valid_points)>1:
                hv = hypervolume((np.asarray(valid_points))).compute(reference_point)
                print("hypervolume " + str(hv))
                filehandle.write(str(hv) + "\n")
            else:
                filehandle.write("0" + "\n" )
        filehandle.close()
