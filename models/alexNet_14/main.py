#!/usr/local/lib/python3.5/dist-packages
#This is the main file for the alexnet14 model
import os
import sys
import json
import matplotlib.pyplot as plt
from alexnet14 import train_model, create_model

#Our datatool
PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(PATH, "..", "..", "python", "tools"))
import datatool

netNum = '14'
run = 0

model = create_model()

Xtr, Ytr, Xte, Yte = datatool.get_training_data(training_ratio=(8/10.0))

score, history = train_model(model, Xtr, Ytr, Xte, Yte,
                             os.path.join(PATH, "train_"+str(run)+".h5"))

plt.plot(history.history['translation_mean_absoulte_error'])
plt.plot(history.history['val_translation_mean_absoulte_error'])
plt.title('model mean_absoulte_error per epoch')
plt.ylabel('translation mean absoulute error')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history.history['rotation_mean_absoulte_error'])
plt.plot(history.history['val_rotation_mean_absoulte_error'])
plt.title('model mean_absoulte_error per epoch')
plt.ylabel('rotation mean absoulute error')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history.history['translation_loss'])
plt.plot(history.history['val_translation_loss'])
plt.title('model translation loss per epoch')
plt.ylabel('rotation loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


plt.plot(history.history['rotation_loss'])
plt.plot(history.history['val_rotation_loss'])
plt.title('model rotation loss per epoch')
plt.ylabel('rotation loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

with open(os.path.join(PATH, "history_"+str(run)+".json"), 'w') as f:
    json.dump(score, f, indent=4)
    json.dump(history, f, indent=4)
