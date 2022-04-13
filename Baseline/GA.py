import os
import time
from shutil import copy, copyfile
import numpy as np
from platypus import NSGAII, Problem, Real, MOEAD, Solution, nondominated, InjectedPopulation
import baseline_simulation_launch

x_tries = []
y_tries = []
const_tries = []


def sim(x):
    global itr_counter
    print(itr_counter)
    original_x, functions, constraints, = baseline_simulation_launch.evaluations(x, itr_counter)
    x_tries.append(original_x)
    y_tries.append(functions)
    const_tries.append(constraints)
    itr_counter += 1
    return functions, constraints


if __name__ == "__main__":

    itr_counter = 0

    paths = '.'

    M = 2  # objectives
    C = 6  # Constraints
    d = 32  # input dimensions
    seed = 1

    np.random.seed(seed)
    total_iterations = 1500
    bound = [0, 1]
    Fun_bounds = [bound] * d
    front = []
    batch_len = 1
    valid_points = []
    functions = [0 for i in range(M)]

    # For using initial.txt file:
    filename = "initial.txt"
    f = open(filename, "r")
    f1 = f.readlines()
    inits = [x.split() for x in f1]
    initial_number = len(inits)
    f.close()

    # filename = "results/Outputs.txt"
    # f = open(filename, "r")
    # f1 = f.readlines()
    # outputs = [x.split() for x in f1]
    # f.close()
    #
    # filename = "results/Original_Inputs.txt"
    # f = open(filename, "r")
    # f1 = f.readlines()
    # orig_inputs = [x.split() for x in f1]
    # f.close()

    vars = []
    objs = []
    consts = []

    problem = Problem(d, M, C)
    init_pop = [Solution(problem) for i in range(initial_number)]
    itr_counter = 0

    for i in range(initial_number):
        x_new = [(float(p)) for p in inits[i]]
        # original_x = [(float(t)) for t in orig_inputs[i]]
        # function_vals = [(float(t)) for t in outputs[i][:M]]
        # constraint_vals = [(float(t)) for t in outputs[i][M + 1:]]
        original_x, function_vals, constraint_vals = baseline_simulation_launch.evaluations(x_new, itr_counter)
        init_pop[i].variables = np.array(original_x)
        init_pop[i].objectives = np.array(function_vals)
        init_pop[i].constraints = np.array(constraint_vals)
        init_pop[i].constraint_violation = 0.0
        init_pop[i].evaluated = True
        itr_counter += 1
    #     print(init_pop[i])
    #
    # for i in init_pop:
    #     print(i)

    problem.types[:] = Real(bound[0], bound[1])
    problem.constraints[:] = [">0", ">=0", ">=0", ">=0", ">=0", ">=0"]
    problem.function = sim
    algorithm = MOEAD(problem, population_size=len(init_pop), generator=InjectedPopulation(init_pop))
    algo_name = 'MOEAD'
    # algorithm = NSGAII(problem, population_size=len(init_pop), generator=InjectedPopulation(init_pop))
    # algo_name='NSGAII'
    algorithm.run(total_iterations)
    cheap_pareto_set = [solution.variables for solution in algorithm.result]
    cheap_pareto_front = [list(solution.objectives) for solution in algorithm.result]

