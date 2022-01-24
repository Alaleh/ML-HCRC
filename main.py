import os
import time
from shutil import copy
import numpy as np
from model import GaussianProcess
from singlemes import MaxvalueEntropySearch
from platypus import NSGAII, Problem, Real
import simulation_launch

#TODO: Try giving it the actual inputs not [0,1]

if __name__ == "__main__":

    paths = '.'

    M = 5  # objectives
    C = 4  # Constraints
    d = 32  # input dimensions
    seed = 1

    np.random.seed(seed)
    total_iterations = 900
    sample_number = 1
    bound = [0, 1]
    Fun_bounds = [bound] * d
    GPs = []
    Multiplemes = []
    GPs_C = []
    Multiplemes_C = []
    front = []
    batch_len = 1
    valid_points = []

    # For using initial.txt file:
    filename = "initial.txt"
    f = open(filename, "r")
    f1 = f.readlines()
    inits = [x.split() for x in f1]
    initial_number = len(inits)
    f.close()

    for i in range(M):
        GPs.append(GaussianProcess(d))
    for i in range(C):
        GPs_C.append(GaussianProcess(d))

    for k in range(initial_number):
        # exist = True
        # while exist:
        #     x_rand = list(np.random.uniform(low=bound[0], high=bound[1], size=(d,)))
        #     if not (any((x_rand == x).all() for x in GPs[0].xValues)):
        #         exist = False
        x_new = [(float(p)) for p in inits[k]]
        # if any((x_rand == x).all() for x in GPs[0].xValues):
        #     print(k, " is duplicate")
        #     continue
        original_x, function_vals, constraint_vals = simulation_launch.evaluations(x_new, k)
        for i in range(M):
            GPs[i].addSample(np.asarray(x_new), np.asarray(function_vals[i]))
        for i in range(C):
            GPs_C[i].addSample(np.asarray(x_new), np.asarray(constraint_vals[i]))
        if constraint_vals[0]>0 and min(constraint_vals[1:])>=0:
            valid_points.append(function_vals)

        with open(os.path.join(paths, 'plotter/results/Inputs.txt'), "a") as filehandle:
            for item in x_new:
                filehandle.write('%f ' % item)
            filehandle.write('\n')
        filehandle.close()
        with open(os.path.join(paths, 'plotter/results/Original_Inputs.txt'), "a") as filehandle:
            for item in original_x:
                filehandle.write('%f ' % item)
            filehandle.write('\n')
        filehandle.close()
        with open(os.path.join(paths, 'plotter/results/Outputs.txt'), "a") as filehandle:
            vals = [x for x in function_vals]
            for x in constraint_vals:
                vals.append(x)
            for listitem in vals:
                filehandle.write('%f ' % listitem)
            filehandle.write('\n')
        filehandle.close()

        copy("hcr_test.ocn", "ocns")
        os.rename("ocns/hcr_test.ocn", "ocns/hcr_test" + "_" + str(k+1) + ".ocn")

    for i in range(M):
        GPs[i].fitModel()
        Multiplemes.append(MaxvalueEntropySearch(GPs[i]))
    for i in range(C):
        GPs_C[i].fitModel()
        Multiplemes_C.append(MaxvalueEntropySearch(GPs_C[i]))

    for l in range(initial_number, total_iterations):

        print(l)

        for i in range(M):
            Multiplemes[i] = MaxvalueEntropySearch(GPs[i])
            Multiplemes[i].Sampling_RFM()

        for i in range(C):
            Multiplemes_C[i] = MaxvalueEntropySearch(GPs_C[i])
            Multiplemes_C[i].Sampling_RFM()

        max_samples = []
        max_samples_constraints = []

        for j in range(sample_number):
            for i in range(M):
                Multiplemes[i].weigh_sampling()
            for i in range(C):
                Multiplemes_C[i].weigh_sampling()

            def CMO(xi):
                xi = np.asarray(xi)
                y = [Multiplemes[i].f_regression(xi)[0][0] for i in range(len(GPs))]
                y_c = [Multiplemes_C[i].f_regression(xi)[0][0] for i in range(len(GPs_C))]
                return y, y_c

            problem = Problem(d, M, C)
            problem.types[:] = Real(bound[0], bound[1])
            problem.constraints[:] = [">0", ">=0", ">=0", ">=0"]
            problem.function = CMO
            algorithm = NSGAII(problem)
            t1 = time.time()
            algorithm.run(3000)
            print("cheap algorithm took : ", time.time() - t1)

            cheap_pareto_front = [list(solution.objectives) for solution in algorithm.result]
            cheap_constraints_values = [list(solution.constraints) for solution in algorithm.result]

            # this is picking the max over the pareto: best case
            maxoffunctions = [-1 * min(f) for f in list(zip(*cheap_pareto_front))]
            maxofconstraints = [-1 * min(f) for f in list(zip(*cheap_constraints_values))]
            max_samples.append(maxoffunctions)
            max_samples_constraints.append(maxofconstraints)

        # TODO add this to GP too -> No need now that we have stability constraint?
        def mesmo_acq(x):
            if np.prod([GPs_C[i].getmeanPrediction(x) >= 0 for i in range(len(GPs_C))]):
                multi_obj_acq_total = 0
                for j in range(sample_number):
                    multi_obj_acq_sample = 0
                    for i in range(M):
                        multi_obj_acq_sample = multi_obj_acq_sample + Multiplemes[i].single_acq(np.asarray(x),max_samples[j][i])
                    for i in range(C):
                        multi_obj_acq_sample = multi_obj_acq_sample + Multiplemes_C[i].single_acq(np.asarray(x),max_samples_constraints[j][i])
                    multi_obj_acq_total = multi_obj_acq_total + multi_obj_acq_sample
                return (multi_obj_acq_total / sample_number)
            else:
                return 10e10


        x_tries = np.random.uniform(bound[0], bound[1],size=(1000, d))
        y_tries=[mesmo_acq(x) for x in x_tries]
        sorted_indecies=np.argsort(y_tries)
        i=0
        x_best=x_tries[sorted_indecies[i]]
        while (any((x_best == x).all() for x in GPs[0].xValues)):
            print(x_best)
            print(GPs[0].xValues)
            i=i+1
            x_best=x_tries[sorted_indecies[i]]

        # ---------------Updating and fitting the GPs-----------------

        original_x, function_vals, constraint_vals = simulation_launch.evaluations(x_best,l)
        for i in range(M):
            GPs[i].addSample(np.asarray(x_best), function_vals[i])
            GPs[i].fitModel()
        for i in range(C):
            GPs_C[i].addSample(np.asarray(x_best), constraint_vals[i])
            GPs_C[i].fitModel()
        if constraint_vals[0]>0 and min(constraint_vals[1:])>=0:
            valid_points.append(function_vals)

        with open(os.path.join(paths, 'plotter/results/Inputs.txt'), "a") as filehandle:
            for item in x_best:
                filehandle.write('%f ' % item)
            filehandle.write('\n')
        filehandle.close()

        with open(os.path.join(paths, 'plotter/results/Original_Inputs.txt'), "a") as filehandle:
            for item in original_x:
                filehandle.write('%f ' % item)
            filehandle.write('\n')
        filehandle.close()

        with open(os.path.join(paths, 'plotter/results/Outputs.txt'), "a") as filehandle:
            vals = [x for x in function_vals]
            for x in constraint_vals:
                vals.append(x)
            for listitem in vals:
                filehandle.write('%f ' % listitem)
            filehandle.write('\n')
        filehandle.close()

        copy("hcr_test.ocn", "ocns")
        os.rename("ocns/hcr_test.ocn", "ocns/hcr_test" + "_" + str(l+1) + ".ocn")
