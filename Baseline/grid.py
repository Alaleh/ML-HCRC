# -*- coding: utf-8 -*-
"""
Created on Thu Aug 22 12:46:42 2019

@author: User
"""

import numpy as np
import baseline_simulation_launch
import os

#grid_sizes=[30,40,50,80,100,200]
grid_sizes=[200]

d=33
seed=1
np.random.seed(seed)
bound=[0,1]
for i in range(len(grid_sizes)):
    x_tries = np.random.uniform(bound[0], bound[1],size=(grid_sizes[i], d))
    y_tries=[baseline_simulation_launch.evaluations(x) for x in x_tries]
    with  open(os.path.join('Results',str(grid_sizes[i])+'Input.txt'), "w") as filehandle:  
        for listitem in x_tries:
            for item in listitem:
                filehandle.write('%f ' % item)
            filehandle.write('\n')
#    filehandle.close()
    with  open(os.path.join('Results',str(grid_sizes[i])+'Front.txt'), "w") as filehandle:  
        for listitem in y_tries:
            for item in listitem:
                filehandle.write('%f ' % item)
            filehandle.write('\n')
#    filehandle.close()
