#!/usr/local/lib/python3.5/dist-packages
#This is the main file for training all the models
import os
import sys
import json
import matplotlib.pyplot as plt
from model import train_model, create_model

#Our datatool
PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(PATH, "..", "python", "tools"))
import datatool

def setup_model(modelType, modelNumber, paramDict):
    netNum = str(modelNumber)
    run = modelNumber
    if(modelType == 'global'):
        path = os.path.join(PATH, "global_models")
    elif(modelType == 'relative'):
        path = os.path.join(PATH, "relative_models")
    elif(modelType == 'quaternion'):
        path = os.path.join(PATH, "quat_models")
    else:
        print ("error: modelType can only be either global, relative, or quaternion")
        return -1;
    
    if(modelType == 'quaternion'):
        model = create_model(paramDict['convolution'], paramDict['dense'], quaternion=True)
    else:
        model = create_model(paramDict['convolution'], paramDict['dense'])
    
    if(modelType == 'global'):
        Xtr, Ytr = datatool.get_training_data(sequences=[0,1,4,5,6,7,8,9,10], training_ratio=(1), no_test=True, no_quaternions=True, global_trans=True)
    elif(modelType == 'relative'):
        Xtr, Ytr = datatool.get_training_data(sequences=[0,1,4,5,6,7,8,9,10] ,training_ratio=(1), no_test=True, no_quaternions=True)
    else:
        Xtr, Ytr = datatool.get_training_data(sequences=[0,1,4,5,6,7,8,9,10] ,training_ratio=(1), no_test=True)
    
    history = train_model(model, Xtr, Ytr, save_path=os.path.join(path, "gen_train_"+str(run)+".h5"))

    with open(os.path.join(path, "gen_history_"+str(run)+".json"), 'w') as f:
        json.dump(history.history, f, indent=4)

# test run the modular models
with open('model_0.json') as data_file:
    params = json.load(data_file)

setup_model('global', 0, params)
setup_model('relative', 0, params)
setup_model('quaternion', 0, params)
