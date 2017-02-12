#!/usr/local/lib/python3.5/dist-packages
import sys
import json
sys.path.append('/home/sexy/source/deep-visual-odometry/python/snippets/')
sys.path.append('/home/sexy/source/deep-visual-odometry/models/')
import formatData
from alexNet_5 import run_model, create_model
netNum = '5'
Xtr, Ytr, Xte, Yte = formatData.knownEnv(formatData.load_data(),formatData.load_poses(), training_ratio=(8/10.0))

run = 0 
score, history = run_model(create_model(), Xtr, Ytr, Xte, Yte, "/home/sexy/source/deep-visual-odometry/models/alexNet_"+netNum+"/train_"+str(run)+".h5")
with open("/home/sexy/source/deep-visual-odometry/models/alexNet_"+netNum+"/history_"+str(run)+".json", 'w') as f:
	json.dump(score, f, indent=4)
	json.dump(history, f, indent=4)

del Xtr
del Ytr
del Xte
del Yte

import separated_datasets