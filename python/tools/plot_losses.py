import json
import numpy as np
import matplotlib.pyplot as plt

def plot_losses(netNum, filename='history_0.json', model_dir='/home/sexy/source/deep-visual-odometry/models/'):
    #Load all data from json
    with open(model_dir+"alexNet_"+str(netNum)+"/"+filename) as data_file:    
        data = json.load(data_file)

    #Set up data for graphing
    losses = [data["loss"],data["val_loss"]]
    num_epoch = [x+1 for x in range(len(losses[0]))]

    #Plot training loss per epoch versus validation loss per epoch
    plt.plot(num_epoch, losses[0], label='Training')
    plt.plot(num_epoch, losses[1], label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()