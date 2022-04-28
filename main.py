import os
import time
import random
import shutil
import copy
import numpy as np
from model import GaussianProcess
from scipy.optimize import minimize as scipyminimize
from singlemes import MaxvalueEntropySearch
from platypus import NSGAII, Problem, Real, Subset
import simulation_launch
from transform import convert_to_original

# 1. We keep adding flags to discredit the good results
# What if the bad results were better?


if __name__ == "__main__":

    paths = '.'

    M = 5  # objectives
    C = 5  # Constraints
    d = 32  # input dimensions
    seed = 22011

    np.random.seed(seed)
    random.seed(seed)

    total_iterations = 2000
    sample_number = 1
    bound = [0, 1]
    Fun_bounds = [bound] * d
    GPs = []
    Multiplemes = []
    GPs_C = []
    Multiplemes_C = []
    front = []
    batch_len = 1
    selected_xs = []
    selected_ys = []

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
    t = []

    filename1 = "prev_res/Original_Inputs_MESMOC_prev.txt"
    f_i = open(filename1, "r")
    f_in = f_i.readlines()
    f_i.close()
    prev_ins = [[float(ss) for ss in z.split()] for z in f_in]

    filename2 = "prev_res/Outputs_MESMOC_prev.txt"
    f_o = open(filename2, "r")
    f_out = f_o.readlines()
    f_o.close()
    prev_outs = [[float(ss) for ss in z.split()] for z in f_out]

    filename3 = "prev_res/Inputs_MESMOC_prev.txt"
    f_n = open(filename3, "r")
    f_norm = f_n.readlines()
    f_n.close()
    prev_inits = [[float(ss) for ss in z.split()] for z in f_norm]

    initial_number = len(prev_ins)
    print(initial_number)

    for k in range(initial_number):

        # if preset initial points:
        #        x_new = [np.round(float(p), 12) for p in inits[k]]
        #        original_x, function_vals, constraint_vals = simulation_launch.evaluations(x_new, k)

        x_new = prev_inits[k]
        original_x, function_vals, constraint_vals = prev_ins[k], prev_outs[k][:M], prev_outs[k][M:]

        for i in range(M):
            GPs[i].addSample(np.asarray(original_x), np.asarray(function_vals[i]))
        for i in range(C):
            GPs_C[i].addSample(np.asarray(original_x), np.asarray(constraint_vals[i]))

        with open(os.path.join(paths, 'results/Inputs_MESMOC.txt'), "a") as filehandle:
            for item in x_new:
                filehandle.write('%f ' % item)
            filehandle.write('\n')
        filehandle.close()
        with open(os.path.join(paths, 'results/Original_Inputs_MESMOC.txt'), "a") as filehandle:
            for item in original_x:
                filehandle.write('%f ' % item)
            filehandle.write('\n')
        filehandle.close()
        with open(os.path.join(paths, 'results/Outputs_MESMOC.txt'), "a") as filehandle:
            vals = [x for x in function_vals]
            for x in constraint_vals:
                vals.append(x)
            for listitem in vals:
                filehandle.write('%f ' % listitem)
            filehandle.write('\n')
        filehandle.close()

        # # If first time running:
        # shutil.copy("hcr_test.ocn", "ocns")
        # os.rename("ocns/hcr_test.ocn", "ocns/hcr_test_MESMOC_" + str(k + 1) + ".ocn")


    for i in range(M):
        GPs[i].fitModel()
        Multiplemes.append(MaxvalueEntropySearch(GPs[i]))
    for i in range(C):
        GPs_C[i].fitModel()
        Multiplemes_C.append(MaxvalueEntropySearch(GPs_C[i]))

    for iter_num in range(initial_number, total_iterations):

        print(iter_num)

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
            ranges = [[15.0, 20.0, 25.0]] * 24 + [
                [4.5, 4.7, 5, 5.6, 6.6, 6.8, 7.5, 8.2, 9.4, 10.0, 12.0, 14.0, 15.0, 17.0, 18.0, 19.0, 20.0,
                 22.0]] * 4 + [[.220, .240, .250, .270, .300, .330, .360, .390, .400, .420, .470, .500, .530, .560, .590,
                                .600, .620, .680, .700, .760, .770, .820, .900, 1, 1.2, 1.5, 1.8, 2.2, 3.3, 4.7]] + [
                         list(range(200, 6000, 100))] + [list(range(200, 3000, 20))] + [list(range(550, 630, 5))]
            for q in range(d):
                problem.types[q] = Subset(ranges[q], 1)
            problem.types[:] = Real(bound[0], bound[1])
            problem.constraints[:] = [">0", ">=0", ">=0", ">=0", ">=0"]
            problem.function = CMO
            algorithm = NSGAII(problem)
            t1 = time.time()
            algorithm.run(4000)
            print("cheap algorithm took : ", time.time() - t1)

            cheap_pareto_front = [list(solution.objectives) for solution in algorithm.result]
            cheap_constraints_values = [list(solution.constraints) for solution in algorithm.result]

            # this is picking the max over the pareto: best case
            maxoffunctions = [-1 * min(f) for f in list(zip(*cheap_pareto_front))]
            maxofconstraints = [-1 * min(f) for f in list(zip(*cheap_constraints_values))]
            max_samples.append(maxoffunctions)
            max_samples_constraints.append(maxofconstraints)


        def mesmo_acq(x11):
            x_t = copy.deepcopy(x11)
            x_n = convert_to_original(x_t)
            preferences = [0.8, 0.05, 0.05, 0.05, 0.05]
            if np.prod([GPs_C[i].getmeanPrediction(x_n) >= 0 for i in range(len(GPs_C))]):
                multi_obj_acq_total = 0
                for j in range(sample_number):
                    multi_obj_acq_sample = 0
                    for i in range(M):
                        multi_obj_acq_sample += preferences[i] * 0.5 * Multiplemes[i].single_acq(np.asarray(x_n),
                                                                                                 max_samples[j][i])
                    for i in range(C):
                        multi_obj_acq_sample += (0.5 / C) * Multiplemes_C[i].single_acq(np.asarray(x_n),
                                                                                        max_samples_constraints[j][i])
                    multi_obj_acq_total = multi_obj_acq_total + multi_obj_acq_sample
                return multi_obj_acq_total / sample_number
            else:
                return 10e10


        # l-bfgs-b acquisation optimization
        x_tries = np.random.uniform(bound[0], bound[1], size=(5000, d))
        y_tries = [mesmo_acq(x) for x in x_tries]
        sorted_indecies = np.argsort(y_tries)
        i = 0
        x_best = list(x_tries[sorted_indecies[i]])

        while x_best in selected_xs:
            print("repeated x")
            i = i + 1
            x_best = list(x_tries[sorted_indecies[i]])
        y_best = y_tries[sorted_indecies[i]]
        x_seed = list(np.random.uniform(low=bound[0], high=bound[1], size=(100, d)))
        for x_try in x_seed:
            result = scipyminimize(mesmo_acq, x0=np.asarray(x_try).reshape(1, -1), method='L-BFGS-B', bounds=Fun_bounds)
            if not result.success:
                continue
            if (result.fun <= y_best) and (not (result.x in np.asarray(GPs[0].xValues))):
                x_best = result.x
                y_best = result.fun

        # ---------------Updating and fitting the GPs-----------------

        original_x, function_vals, constraint_vals = simulation_launch.evaluations(x_best, iter_num)
        for i in range(M):
            GPs[i].addSample(np.asarray(original_x), function_vals[i])
            GPs[i].fitModel()
        for i in range(C):
            GPs_C[i].addSample(np.asarray(original_x), constraint_vals[i])
            GPs_C[i].fitModel()

        selected_xs.append(list(x_best))

        with open(os.path.join(paths, 'results/Inputs_MESMOC.txt'), "a") as filehandle:
            for item in x_best:
                filehandle.write('%f ' % item)
            filehandle.write('\n')
        filehandle.close()

        with open(os.path.join(paths, 'results/Original_Inputs_MESMOC.txt'), "a") as filehandle:
            for item in original_x:
                filehandle.write('%f ' % item)
            filehandle.write('\n')
        filehandle.close()

        with open(os.path.join(paths, 'results/Outputs_MESMOC.txt'), "a") as filehandle:
            vals = [x for x in function_vals]
            for x in constraint_vals:
                vals.append(x)
            for listitem in vals:
                filehandle.write('%f ' % listitem)
            filehandle.write('\n')
        filehandle.close()

        shutil.copy("hcr_test.ocn", "ocns")
        os.rename("ocns/hcr_test.ocn", "ocns/hcr_test_MESMOC_" + str(iter_num + 1) + ".ocn")
