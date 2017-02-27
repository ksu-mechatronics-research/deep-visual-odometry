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

netNum = '0'
run = 0

model = create_model()

Xtr, Ytr = datatool.get_training_data(sequences=[0,1,4,5,6,7,8,9,10] ,training_ratio=(1), no_test=True, no_quaternions=True)

history = train_model(model, Xtr, Ytr, save_path=os.path.join(PATH, "train_"+str(run)+".h5"))

with open(os.path.join(PATH, "history_"+str(run)+".json"), 'w') as f:
    json.dump(history, f, indent=4)
