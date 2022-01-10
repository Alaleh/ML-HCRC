import math

import matplotlib.pyplot as plt
import numpy as np
import os
import re

ls = os.listdir('hpvs/')
file_names = [x for x in ls if x.endswith('.txt')]
print(file_names)

All_phv_algo = []
leg = [x[13:-4] for x in file_names]

for file in file_names:
    f = open('../hpvs/' + file, "r")
    print(f)
    f1 = f.readlines()
    hypers = [float(i.split()[0]) for i in f1]
    All_phv_algo.append(hypers)

print([len(i) for i in All_phv_algo])

max_iters = max([len(j) for j in All_phv_algo])
print(max_iters)

plt.rcParams.update({'font.size': 10})
plt.rcParams['figure.figsize'] = 8, 5

color = ['#9c031f', 'k', '#ff1500', '#e37e30', 'darkslategrey', 'c', 'k', 'y']
markers = ["+", "*", "x", "v", ".", "^", "-"]

# Scaled and unscaled are used for normalized hypervolumes

for scale in ["unscaled"]:  # you can also add "zoomed", "scaled", , "unscaled"

    plt.grid(b=True, linestyle='--')
    fig1 = plt.figure(1)
    plt.xlim(1, max_iters + 1)
    plt.ylim(1.4,1.85)

    for i in range(len(All_phv_algo)):
        plt.plot(range(len(All_phv_algo[i])), All_phv_algo[i], label=leg[i], linewidth=1.5, color=color[i],
                 marker=markers[i], markevery=10)

    plt.legend(loc='lower right', prop={'size': 6})  # bbox_to_anchor=(1.01, 1),

    plt.title(" Hypervolumes Plot")
    plt.xlabel('Number of simulations (t)', fontsize=15)
    plt.ylabel('Hypervolume Indicator', fontsize=15)
    plt.show()
    fig1.savefig('../hpvs/plots/hpv_plot.png')

    plt.close()
