import json
import numpy as np
import matplotlib.pyplot as plt

def plot_losses(netNum, filename='history_0.json', model_dir='/home/sexy/source/deep-visual-odometry/models/'):
    #Load all data from json
    with open(model_dir+"alexNet_"+str(netNum)+"/"+filename) as data_file:    
        data = json.load(data_file)

    #Set up data for graphing
    losses = [data["loss"],data["val_loss"]]
    num_epoch = [x for x in range(len(losses[0]))]

    #Plot loss per epoch versus validation loss per epoch
    plt.plot(num_epoch, losses[0], num_epoch, losses[1]) 
    plt.show()