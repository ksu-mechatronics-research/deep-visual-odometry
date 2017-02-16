#!/usr/local/lib/python3.5/dist-packages
#This is the main file for the alexnet14 model
import os
import sys
import json
#from alexnet14 import run_model, create_model
PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(PATH,"..","..","python","tools"))
import formatData

netNum = '14'
run = 0

model = create_model()

Xtr, Ytr, Xte, Yte = formatData.knownEnv(formatData.load_data(),formatData.load_poses(), training_ratio=(8/10.0))



score, history = run_model(model, Xtr, Ytr, Xte, Yte,
                           os.path.join(PATH, "train_"+str(run)+".h5"))
with open(os.path.join(PATH, "history_"+str(run)+".json"), 'w') as f:
    json.dump(score, f, indent=4)
    json.dump(history, f, indent=4)
