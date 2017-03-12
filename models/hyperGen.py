#!/usr/local/lib/python3.5/dist-packages
import os
import sys
import json
import random
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
random.seed(datetime.now())

PATH = os.path.dirname(os.path.abspath(__file__))

kernalNums = []
for i in range(64, 361):
    for ii in range(360-i):
        kernalNums.append(i)

"""
array = np.zeros(10000)
for i in range(10000):
    array[i] = random.choice(kernalNums)
plt.hist(array, bins=np.arange(array.min(), array.max()+1))
plt.show()
"""

for j in range(3, 50):
    activations = ["prelu"]*5 + ["relu"]*3 + ["tanh"]*2
    poolings = [3]*8 + [5]*3 + [7]*2

    output_dict = {}

    convolutions = []
    convolutionLayers = random.randint(2, 4)
    for i in range(convolutionLayers):
        kernalNum = random.choice(kernalNums)
        kernalSize = 2*random.randint(1, 5)+1
        activation = random.choice(activations)
        poolSize = random.choice(poolings)
        poolStride = 2*random.randint(1, (poolSize-1)/2)+1
        convolutions.append([kernalNum, kernalSize, activation, poolSize, poolStride])
    dense = []
    denseLayers = random.randint(1, 4)
    denseSizes = []
    for i in range(denseLayers):
        denseSizes.append(random.randint(5, 12))
    denseSizes.sort()
    denseSizes = denseSizes[::-1]
    for i in range(denseLayers):
        denseSize = 2**denseSizes[i]
        activation = random.choice(activations)
        dense.append([denseSize, activation])
    output_dict['convolution'] = convolutions
    output_dict['dense'] = dense

    #print(output_dict)
    with open(os.path.join(PATH, "gen_models", "model_"+str(j)+".json"), 'w') as f:
        json.dump(output_dict, f, indent=4)
