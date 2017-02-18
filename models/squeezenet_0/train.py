#!/usr/local/lib/python3.5/dist-packages
#This is the training file for the squeezenet0 model
import os
import sys
import json
import matplotlib.pyplot as plt
from model import train_model, create_model
PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(PATH,"..","..","python","tools"))
import formatData

netNum = '0'
run = 0

model = create_model()

#Use sequences 0-9
sequences = []
for i in range(10):
    sequences.append(i)
    
Xtr, Ytr, Xte, Yte = formatData.get_training_data(sequences, training_ratio=(1))

score, history = train_model(model, Xtr, Ytr, Xte, Yte,
                           os.path.join(PATH, "train_"+str(run)+".h5"))

with open(os.path.join(PATH, "history_"+str(run)+".json"), 'w') as f:
    json.dump(score, f, indent=4)
    json.dump(history, f, indent=4)

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
