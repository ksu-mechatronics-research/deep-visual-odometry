#!/usr/local/lib/python3.5/dist-packages
import sys
import json
repo_dir = "/home/sexy/source/deep-visual-odometry/"
sys.path.append(repo_dir+'python/snippets/')
sys.path.append(repo_dir+'models/')

import formatData
from alexNet_5 import run_model, create_model
netNum = '5'
path = repo_dir + "models/alexNet_"+netNum+"/"

score = [[] for i in range(10)]
history = [{} for i in range(10)]
run = 1 

model = create_model()
for i in range(10):
    Xtr, Ytr, Xte, Yte = formatData.knownEnv(formatData.load_data(),formatData.load_poses(),sequences=[i], training_ratio=(8/10.0))
    score[i], temp_hist[i] = run_model(model, Xtr, Ytr, Xte, Yte)

    
model.save("train_"+str(run)+".h5")

with open(path+"history_"+str(run)+".json", 'w') as f:
    json.dump(score, f, indent=4)
    json.dump(history, f, indent=4)