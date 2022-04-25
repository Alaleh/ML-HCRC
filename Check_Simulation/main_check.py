import os
import time
import random
from shutil import copy
import numpy as np
import retest_simulation_launch
import retest_rerun_simulation
import retest_longer_simulation

# TODO: Try giving it the actual inputs not [0,1]

if __name__ == "__main__":

    paths = '.'

    M = 2  # objectives
    C = 6  # Constraints
    d = 32  # input dimensions
    seed = 11

    np.random.seed(seed)
    random.seed(seed)

    # [0,1] values
    filename1 = "inputs/test_points.txt"
    f_i = open(filename1, "r")
    f_in = f_i.readlines()
    f_i.close()
    input_tests = [[float(ss) for ss in z.split()] for z in f_in]
    initial_number = len(input_tests)

    for k in range(initial_number):

        original_x1, function_vals1, constraint_vals1 = retest_simulation_launch.evaluations(input_tests[k], k)

        with open(os.path.join(paths, 'results/Original_Inputs1.txt'), "a") as filehandle:
            for item in original_x1:
                filehandle.write('%f ' % item)
            filehandle.write('\n')
        filehandle.close()

        with open(os.path.join(paths, 'results/sim1.txt'), "a") as filehandle:
            for item in function_vals1:
                filehandle.write('%f ' % item)
            filehandle.write('\n')
        filehandle.close()

        with open(os.path.join(paths, 'results/const1.txt'), "a") as filehandle:
            for item in constraint_vals1:
                filehandle.write('%f ' % item)
            filehandle.write('\n')
        filehandle.close()

        copy("hcr_test.ocn", "ocns")
        os.rename("ocns/hcr_test.ocn", "ocns/hcr_test_1s_" + "_" + str(k + 1) + ".ocn")

        original_x2, function_vals2, constraint_vals2 = retest_rerun_simulation.re_evaluations(input_tests[k], k)

        with open(os.path.join(paths, 'results/Original_Inputs2.txt'), "a") as filehandle:
            for item in original_x2:
                filehandle.write('%f ' % item)
            filehandle.write('\n')
        filehandle.close()

        with open(os.path.join(paths, 'results/sim2.txt'), "a") as filehandle:
            for item in function_vals2:
                filehandle.write('%f ' % item)
            filehandle.write('\n')
        filehandle.close()

        with open(os.path.join(paths, 'results/const2.txt'), "a") as filehandle:
            for item in constraint_vals2:
                filehandle.write('%f ' % item)
            filehandle.write('\n')
        filehandle.close()

        copy("hcr_test.ocn", "ocns")
        os.rename("ocns/hcr_test.ocn", "ocns/hcr_test_2s_" + "_" + str(k + 1) + ".ocn")

        original_x3, function_vals3, constraint_vals3 = retest_longer_simulation.re_re_evaluations(input_tests[k], k)

        with open(os.path.join(paths, 'results/Original_Inputs3.txt'), "a") as filehandle:
            for item in original_x3:
                filehandle.write('%f ' % item)
            filehandle.write('\n')
        filehandle.close()

        with open(os.path.join(paths, 'results/sim3.txt'), "a") as filehandle:
            for item in function_vals3:
                filehandle.write('%f ' % item)
            filehandle.write('\n')
        filehandle.close()

        with open(os.path.join(paths, 'results/const3.txt'), "a") as filehandle:
            for item in constraint_vals3:
                filehandle.write('%f ' % item)
            filehandle.write('\n')
        filehandle.close()

        copy("hcr_test.ocn", "ocns")
        os.rename("ocns/hcr_test.ocn", "ocns/hcr_test_3s_" + "_" + str(k + 1) + ".ocn")



