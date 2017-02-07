# Y axis is elevation
from os import listdir
import numpy as np
import pykitti
import matplotlib.pyplot as plt
from scipy.misc import imread

size = 128

def load_data():
    directory = "/home/sexy/Documents/dataset/processed_imgs_128/sequence"
    list_X = []
    for i in range(11):
        dirs = listdir(directory+str(i))
        dirs.sort()
        array = np.zeros((len(dirs), size, size, 3),dtype='uint8')
        for j, dirj in enumerate(dirs):
            array[j,:,:,:] = imread(directory+str(i)+"/"+dirj)
        list_X.append(array)
    return list_X

def load_poses():
    directoryL = "/home/sexy/Documents/dataset/"
    list_Y = []
    for i in range(11):
        data = pykitti.odometry(directoryL, str(i).zfill(2))
        data.load_poses()
        Y = np.zeros((len(data.T_w_cam0),3))
        for j in range(len(data.T_w_cam0)):
            Y[j,:] = data.T_w_cam0[j][:-1,-1]
        delta_Y = np.zeros((len(data.T_w_cam0),3))
        delta_Y[1:,:] = Y[1:,:]-Y[:-1,:]
        list_Y.append(delta_Y)
    return list_Y

def knownEnv(data, poses, sequences = [0,1,2,3,4,5,6,7,8,9,10], training_ratio = (4/5.0)):
    ind = []
    ind_total = []
    for i in sequences:
        ind_total.append(np.size(data[i],axis=0))
        ind.append(int(ind_total[i]*4/5.0))
    Xtr = np.zeros((sum(ind)-len(sequences),size,size,6), dtype="uint8")
    Ytr = np.zeros((sum(ind)-len(sequences),3))
    Xte = np.zeros((sum(ind_total)-sum(ind),size,size,6), dtype="uint8")
    Yte = np.zeros((sum(ind_total)-sum(ind),3))
    
    countTr = 0
    countTe = 0
    for i in sequences:
        Xtr[countTr:countTr+ind[i]-1,:,:,:3] = data[i][:ind[i]-1,:,:,:] #0 -> not inclusive 
        Xtr[countTr:countTr+ind[i]-1,:,:,3:] = data[i][1:ind[i],:,:,:]
        
        Ytr[countTr:countTr+ind[i]-1,:] = poses[i][1:ind[i],:]
        
        #0 -> total-80%
        Xte[countTe:countTe+(ind_total[i]-ind[i]),:,:,:3] = data[i][(ind[i]-1):-1,:,:,:]
        Xte[countTe:countTe+(ind_total[i]-ind[i]),:,:,3:] = data[i][ind[i]:,:,:,:]
        
        Yte[countTe:countTe+(ind_total[i]-ind[i]),:] = poses[i][ind[i]:,:]
        
        countTr += ind[i]-1
        countTe += (ind_total[i]-ind[i])
    
    return Xtr, Ytr, Xte, Yte

    
