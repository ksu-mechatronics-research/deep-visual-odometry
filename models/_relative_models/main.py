#!/usr/local/lib/python3.5/dist-packages
#This is the main file for the alexnet14 model
import os
import sys
import json
import matplotlib.pyplot as plt
from model_0 import train_model, create_model

#Our datatool
PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(PATH, "..", "..", "python", "tools"))
import datatool

netNum = '7'
run = 0

model = create_model()

Xtr, Ytr, Xte, Yte = datatool.get_training_data(training_ratio=(1))

history = run_model(create_model(), Xtr, Ytr, Xte, Yte, "/home/sexy/source/deep-visual-odometry/models/alexNet_"+netNum+"/train_"+str(run)+".h5")


score, history = run_model(create_model(), Xtr, Ytr, Xte, Yte, "/home/sexy/source/deep-visual-odometry/models/alexNet_"+netNum+"/train_"+str(run)+".h5")
with open("/home/sexy/source/deep-visual-odometry/models/alexNet_"+netNum+"/history_"+str(run)+".json", 'w') as f:
    json.dump(score, f, indent=4)
    json.dump(history, f, indent=4)
