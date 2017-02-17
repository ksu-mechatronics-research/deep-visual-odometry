# Y axis is elevation
from os import listdir
import quaternion as q
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
    list_Y = [[],[]]
    for i in range(11):
        data = pykitti.odometry(directoryL, str(i).zfill(2))
        data.load_poses()

        #Empty arrays to hold data
        Translations = np.zeros((len(data.T_w_cam0),3))
        Quaternions = np.zeros(len(data.T_w_cam0), dtype=q.quaternion)

        #store quaternions and translations
        for j in range(len(data.T_w_cam0)):
            Translations[j,:] = data.T_w_cam0[j][:-1,-1]
            Quaternions[j] = (q.from_rotation_matrix(data.T_w_cam0[j][:-1,:-1])).normalized()

        # Make empty arrays to hold data
        delta_trans = np.zeros((len(data.T_w_cam0),3))
        corrected_delta_trans = np.zeros((len(data.T_w_cam0),3))
        delta_quat = np.zeros(len(data.T_w_cam0),dtype=q.quaternion)

        #Get translations between frames
        delta_trans[:-1,:] = Translations[1:,:]-Translations[:-1,:]

        #Correct translations to provide relative motion from car's perspective
        for k in range(len(delta_trans)-1):
            corrected_delta_trans[k] = (Quaternions[k]*q.quaternion(0.,*delta_trans[k])*Quaternions[k].conj()).vec

        #get rotations between frames
        delta_quat[:-1] = Quaternions[:-1].conj()*Quaternions[1:]

        list_Y[0].append(corrected_delta_trans[:-1])
        list_Y[1].append(delta_quat[:-1])
    return list_Y

def knownEnv(data, poses, sequences=[0,1,2,3,4,5,6,7,8,9,10], training_ratio = (4/5.0)):
    ind = []
    ind_total = []
    for i in sequences:
        ind_total.append(np.size(data[i],axis=0))
        ind.append(int(ind_total[i]*training_ratio))
    Xtr = np.zeros((sum(ind)-len(sequences),size,size,6), dtype="uint8")
    Ytr = np.zeros((sum(ind)-len(sequences),3))
    Xte = np.zeros((sum(ind_total)-sum(ind),size,size,6), dtype="uint8")
    Yte = np.zeros((sum(ind_total)-sum(ind),3))
    
    countTr = 0
    countTe = 0
    for i in sequences:
        Xtr[countTr:countTr+ind[i]-1,:,:,:3] = data[i][:ind[i]-1,:,:,:] #0 -> not inclusive 
        Xtr[countTr:countTr+ind[i]-1,:,:,3:] = data[i][1:ind[i],:,:,:]
        
        Ytr[countTr:countTr+ind[i]-1,:] = poses[0][i][:ind[i]-1,:]
        
        #0 -> total-80%
        Xte[countTe:countTe+(ind_total[i]-ind[i]),:,:,:3] = data[i][(ind[i]-1):-1,:,:,:]
        Xte[countTe:countTe+(ind_total[i]-ind[i]),:,:,3:] = data[i][ind[i]:,:,:,:]
        
        Yte[countTe:countTe+(ind_total[i]-ind[i]),:] = poses[0][i][ind[i]-1:,:]
        
        countTr += ind[i]-1
        countTe += (ind_total[i]-ind[i])
    
    return Xtr, Ytr, Xte, Yte

    
