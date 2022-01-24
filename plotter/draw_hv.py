import math

import matplotlib.pyplot as plt
import numpy as np
import os
import re

ls = os.listdir('../plotter/hvs/')
file_names = [x for x in ls if x.endswith('.txt')]
print(file_names)

All_phv_algo = []
leg = [x[13:-4] for x in file_names]

for file in file_names:
    print(file)
    f = open('../plotter/hvs/' + file, "r")
    f1 = f.readlines()
    hypers = [float(i.split()[0]) for i in f1]
    All_phv_algo.append(hypers)

print([len(i) for i in All_phv_algo])

max_iters = 350 #max([len(j) for j in All_phv_algo])
print(max_iters)

plt.rcParams.update({'font.size': 10})
plt.rcParams['figure.figsize'] = 8, 5

color = ["red", "#d9a5b3", "#1868ae", "green", '#9c031f', 'k', '#ff1500', '#e37e30', 'darkslategrey', 'c', 'k', 'y']
markers = ["+", "*", "x", "v", ".", "^", "-"]

# Scaled and unscaled are used for normalized hypervolumes

plt.grid(visible=True, linestyle='--')
fig1 = plt.figure(1)
plt.xlim(1, max_iters + 1)
plt.ylim(14,16.5)

for i in range(len(All_phv_algo)):
    plt.plot(range(len(All_phv_algo[i])), All_phv_algo[i], label=leg[i], linewidth=1.5, color=color[i],
             marker=markers[i], markevery=10)

plt.legend(loc='lower right', prop={'size': 6})  # bbox_to_anchor=(1.01, 1),

plt.title(" Hypervolumes Plot")
plt.xlabel('Number of feasible simulations (t)', fontsize=15)
plt.ylabel('Hypervolume Indicator', fontsize=15)
plt.show()
fig1.savefig('../plotter/plots/hv_plot.png')

plt.close()
