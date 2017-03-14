import numpy as np
import os
import json
import pykitti
from scipy.misc import imread, imresize
import matplotlib.pyplot as plt
from keras.models import load_model
import formatData

# loads a model and runs a forward pass through either entire sequence, train, or test
# output = (0 == entire sequence) (1 == train data) (2 == test data)
def plot_model(netNum = '3', which_model = 'gen_train_', tr_ratio = (10/10.0)):
    model_dir = '/home/sexy/source/deep-visual-odometry/models/'

    model = load_model(model_dir+"global_models/"+which_model+netNum+".h5")

    image_data = formatData.load_data()
    pose_data = formatData.load_poses()
    Xtr, Ytr, Xte, Yte = formatData.knownEnv(image_data, pose_data, training_ratio=tr_ratio)

    # split training and tests into each sequence
    Xtr_list = []
    Ytr_list = []
    Xte_list = []
    Yte_list = []
    for i in range(11):
        _Xtr, _Ytr, _Xte, _Yte = formatData.knownEnv(image_data, pose_data, sequences=[i], training_ratio=tr_ratio)
        Xtr_list.append(_Xtr)
        Ytr_list.append(_Ytr)
        Xte_list.append(_Xte)
        Yte_list.append(_Yte)
        print("Xtr_list[" + str(i) + "] = " + str(Xtr_list[i].shape))
        print("Ytr_list[" + str(i) + "] = " + str(Ytr_list[i].shape))
        print("Xte_list[" + str(i) + "] = " + str(Xte_list[i].shape))
        print("Yte_list[" + str(i) + "] = " + str(Yte_list[i].shape))

    print("Xtr = " + str(Xtr.shape))
    print("Ytr = " + str(Ytr.shape))
    print("Xte = " + str(Xte.shape))
    print("Yte = " + str(Yte.shape))

    predicted_outputs = [[],[],[]]
    plot_data = [[],[],[]]

    # run through the model for each sequence
    for i in range(11):
        # create a temp array with all of training and test of sequence i
        dim1 = Xtr_list[i][:,0,0,0].shape[0]+Xte_list[i][:,0,0,0].shape[0]
        temp_s = np.zeros((dim1, 128, 128, 6))
        temp_s[:Xtr_list[i][:,0,0,0].shape[0]] = Xtr_list[i]
        temp_s[Xtr_list[i][:,0,0,0].shape[0]:] = Xte_list[i]
        predicted_outputs[0].append(model.predict_proba(temp_s, batch_size=4))
        predicted_outputs[1].append(model.predict_proba(Xtr_list[i], batch_size=4))
        predicted_outputs[2].append(model.predict_proba(Xte_list[i], batch_size=4))

    # Sets up the data for the ploting of X vs Z
    for i in range(11):
        temp_Ys = np.concatenate((Ytr_list[i],Yte_list[i]), axis=0)
        print(temp_Ys.shape)
        plot_data[0].append(np.zeros((temp_Ys.shape[0],4)))
        plot_data[1].append(np.zeros((Ytr_list[i].shape[0],4)))
        plot_data[2].append(np.zeros((Yte_list[i].shape[0],4)))
        for j in range(temp_Ys.shape[0]):
            plot_data[0][i][j,0] = np.sum(temp_Ys[:j,0])
            plot_data[0][i][j,1] = np.sum(temp_Ys[:j,2])
            plot_data[0][i][j,2] = np.sum(predicted_outputs[0][i][:j,0])
            plot_data[0][i][j,3] = np.sum(predicted_outputs[0][i][:j,2])
        for j in range(Ytr_list[i].shape[0]):
            plot_data[1][i][j,0] = np.sum(Ytr_list[i][:j,0])
            plot_data[1][i][j,1] = np.sum(Ytr_list[i][:j,2])
            plot_data[1][i][j,2] = np.sum(predicted_outputs[1][i][:j,0])
            plot_data[1][i][j,3] = np.sum(predicted_outputs[1][i][:j,2])
        for j in range(Yte_list[i].shape[0]):
            plot_data[2][i][j,0] = np.sum(Yte_list[i][:j,0])
            plot_data[2][i][j,1] = np.sum(Yte_list[i][:j,2])
            plot_data[2][i][j,2] = np.sum(predicted_outputs[2][i][:j,0])
            plot_data[2][i][j,3] = np.sum(predicted_outputs[2][i][:j,2])
    
    # plots the result
    plt.figure(1, figsize=(16,12))
    for i in range(11):
        plt.subplot(3,4,i+1)
        plt.plot(plot_data[0][i][:,0], plot_data[0][i][:,1], plot_data[0][i][:,2], plot_data[0][i][:,3])
    
    plt.figure(2, figsize=(16,12))
    for i in range(11):
        plt.subplot(3,4,i+1)
        plt.plot(plot_data[1][i][:,0], plot_data[1][i][:,1], plot_data[1][i][:,2], plot_data[1][i][:,3])

    plt.figure(3, figsize=(16,12))
    for i in range(11):
        plt.subplot(3,4,i+1)
        plt.plot(plot_data[2][i][:,0], plot_data[2][i][:,1], plot_data[2][i][:,2], plot_data[2][i][:,3])

    #plt.show()
    
    # save models evaluation foward pass
    with open(model_dir+"alexNet_"+netNum+"/eval_S.json", 'w') as f:
        temp_plot = []
        for i in range(11):
            temp_plot.append(plot_data[0][i].tolist())
        json.dump(temp_plot, f, indent=4)
    with open(model_dir+"alexNet_"+netNum+"/eval_Tr.json", 'w') as f:
        temp_plot = []
        for i in range(11):
            temp_plot.append(plot_data[1][i].tolist())
        json.dump(temp_plot, f, indent=4)
    with open(model_dir+"alexNet_"+netNum+"/eval_Te.json", 'w') as f:
        temp_plot = []
        for i in range(11):
            temp_plot.append(plot_data[2][i].tolist())
        json.dump(temp_plot, f, indent=4)

plot_model()
plot_model(netNum = '4')
plot_model(netNum = '5')
