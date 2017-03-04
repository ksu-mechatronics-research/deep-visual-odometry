#This is the training file for the quaternion model
import os
import sys
import json
import matplotlib.pyplot as plt
from keras.optimizers import Adam
import model
PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(PATH,"..","..","python","tools"))
import datatool

def train_model(MODEL, x_tr, y_tr, x_te, y_te, save_path=None):
    "Note: y should be [[translation],[quat rotation]]"

    MODEL.compile(loss='mean_squared_error', 
                optimizer=Adam(lr=0.006, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0),
                metrics=['mean_absolute_error'])

    history = MODEL.fit(x_tr, y_tr, validation_split=0.2, batch_size=32, nb_epoch=10, verbose=1)

    score = MODEL.evaluate(x_te, y_te, verbose=1)

    if save_path:
        MODEL.save(save_path)

    return score, history

net_num = '0'
run = 0

MODEL = model.create_model()

#Use sequences 0-9
sequences = []
for i in range(10):
    sequences.append(i)

x_tr, y_tr, x_te, y_te = datatool.get_training_data(sequences, training_ratio=(1))

score, history = train_model(MODEL, x_tr, y_tr, x_te, y_te,
                             os.path.join(PATH, "train_"+str(run)+".h5"))

with open(os.path.join(PATH, "history_"+str(run)+".json"), 'w') as f:
    json.dump(score, f, indent=4)
    json.dump(history, f, indent=4)


