# Y axis is elevation
import os
import quaternion as q
import numpy as np
import pykitti
import matplotlib.pyplot as plt
from scipy.misc import imread

def load_images(sequences=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], img_root='processed_imgs_128'):
    '''
    Load images from KITTI dataset. Pass in dir to change the images to use.

    Gets images from:

        deep-visual-odometry/dataset/<img_root>/sequence(0-11)

    args:

        sequences:

            sequences to grab images from (defaults 0-11)

        img_root:

            folder to grab images from

    returns:

        list of 11 lists,

        if i in sequences:

            list[i] = np.array(<images in sequence i>,x,y,image_channels)

        if i not in sequences:

            list[i] = #empty list

    Ex: dir = 'processed_imgs_128'

    gets data from:

        deep-visual-odometry/dataset/processed_imgs_128/sequence#
    '''

    #origional: images_directory = "/home/sexy/Documents/dataset/processed_imgs_128/sequence"
    images_directory = os.path.join('..', '..', 'dataset', img_root, 'sequence')

    list_x = []

    #read number of channels in image
    test_image_dim = imread(os.path.join(images_directory+'0',
                                         os.listdir(images_directory+'0')[0])).shape
    image_channels = test_image_dim[2]
    image_x = test_image_dim[0]
    image_y = test_image_dim[1]

    for i in range(11):
        if i in sequences:
            #get and append image data
            dirs = os.listdir(images_directory+str(i))
            dirs.sort()
            array = np.zeros((len(dirs), image_x, image_y, image_channels), dtype='uint8')
            for j, dirj in enumerate(dirs):
                array[j, :, :, :] = imread(os.path.join(images_directory+str(i), dirj))
            list_x.append(array)
        else:
            #append empty array if not in sequences
            list_x.append([])
    return list_x

def load_poses(sequences=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]):
    '''
    Load poses from KITTI dataset.

    Directory Structure (kitta dataset):

        deep-visual-odometry/dataset/

    args:

        sequences:

            sequences to grab poses from (defaults 0-11)

    returns:

        list of 2 list, each containing 11 lists (list[0,1][1,...,11],

        if i in sequences:

            list[0][i] = relative delta translation (numpy array (1x3))

            list[1][i] = relative quaternion rotation (numpy array (1x4))

        if i not in sequences:

            list[0][i] = empty list

            list[1][i] = empty list
    '''
    # original: poses_directory = "/home/sexy/Documents/dataset/"
    poses_directory = os.path.join('..', '..', 'dataset')
    list_y = [[], []]
    for i in range(11):
        if i in sequences:
            #Get and append odometry data
            data = pykitti.odometry(poses_directory, str(i).zfill(2))
            data.load_poses()

            #Empty arrays to hold data
            translations = np.zeros((len(data.T_w_cam0), 3))
            quaternions = np.zeros(len(data.T_w_cam0), dtype=q.quaternion)
            delta_trans = np.zeros((len(data.T_w_cam0), 3))
            relative_delta_trans = np.zeros((len(data.T_w_cam0), 3))
            delta_quat = np.zeros(len(data.T_w_cam0), dtype=q.quaternion)

            #store quaternions and translations
            for j in range(len(data.T_w_cam0)):
                translations[j, :] = data.T_w_cam0[j][:-1, -1]
                quaternions[j] = (q.from_rotation_matrix(data.T_w_cam0[j][:-1, :-1])).normalized()

            #Get translations between frames
            delta_trans[:-1, :] = translations[1:, :]-translations[:-1, :]

            #Correct translations to provide relative motion from car's perspective
            for k in range(len(delta_trans)-1):
                relative_delta_trans[k] = (quaternions[k]* q.quaternion(0., *delta_trans[k])
                                           *quaternions[k].conj()).vec

            #get rotations between frames
            delta_quat[:-1] = quaternions[1:]/quaternions[:-1]

            list_y[0].append(relative_delta_trans[:-1])
            list_y[1].append(q.as_float_array(delta_quat[:-1]))
        else:
            #append empty sequence if not in sequences
            list_y[0].append([])
            list_y[1].append([])
    return list_y

def get_training_data(sequences=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], training_ratio=(0.8), image_dir='', seperate_images=False):
    '''
    get training data from the KITTI dataset

    args:

        sequences:

            List of training sequences to use

        training_ratio:

            fraction of available data to include in train (default to 4/5)

        image_dir:

            directory to load images from

    returns:

        x_tr:

            train imput data

        y_tr:

            train output data: [0]=translations, [1]=rotations

        x_te:

            test inputs

        y_te:

            test outputs: [0]=translations, [1]=rotations
    '''

    ind = []
    ind_total = []
    y_tr = [[], []]
    y_te = [[], []]

    #load data
    if image_dir:
        images = load_images(sequences, image_dir)
    else:
        images = load_images(sequences)
    poses = load_poses(sequences)

    #set dimenstions of images
    image_channels = images[sequences[0]].shape[3]
    image_size_y = images[sequences[0]].shape[2]
    image_size_x = images[sequences[0]].shape[1]
    #get indices for data slicing
    for i in range(11):
        if i in sequences:
            ind_total.append(np.size(images[i], axis=0))
            ind.append(int(ind_total[i]*training_ratio))
        else:
            ind_total.append(0)
            ind.append(0)

    #Init with empty matrices
    if seperate_images:
        x_tr = [[],[]]
        x_te = [[],[]]
        x_tr[0] = np.zeros((sum(ind)-len(sequences), image_size_x,
                        image_size_y, image_channels), dtype="uint8")
        x_tr[1] = np.zeros((sum(ind)-len(sequences), image_size_x,
                        image_size_y, image_channels), dtype="uint8")
        x_te[0] = np.zeros((sum(ind_total)-sum(ind), image_size_x,
                        image_size_y, image_channels), dtype="uint8")
        x_te[1] = np.zeros((sum(ind_total)-sum(ind), image_size_x,
                        image_size_y, image_channels), dtype="uint8")
    else:
        x_tr = np.zeros((sum(ind)-len(sequences), image_size_x,
                        image_size_y, image_channels*2), dtype="uint8")
        x_te = np.zeros((sum(ind_total)-sum(ind), image_size_x,
                        image_size_y, image_channels*2), dtype="uint8")
    y_tr[0] = np.zeros((sum(ind)-len(sequences), 3))
    y_tr[1] = np.zeros((sum(ind)-len(sequences), 4))
    y_te[0] = np.zeros((sum(ind_total)-sum(ind), 3))
    y_te[1] = np.zeros((sum(ind_total)-sum(ind), 4))

    count_tr = 0
    count_te = 0
    for i in sequences:

        #Get training data for sequence i
        if seperate_images:
            x_tr[0][count_tr:count_tr+ind[i]-1, :, :, :] = images[i][:ind[i]-1, :, :, :]
            x_tr[1][count_tr:count_tr+ind[i]-1, :, :, :] = images[i][1:ind[i], :, :, :]
        else:
            x_tr[count_tr:count_tr+ind[i]-1, :, :, :image_channels] = images[i][:ind[i]-1, :, :, :]
            x_tr[count_tr:count_tr+ind[i]-1, :, :, image_channels:] = images[i][1:ind[i], :, :, :]

        y_tr[0][count_tr:count_tr+ind[i]-1, :] = poses[0][i][:ind[i]-1, :]
        y_tr[1][count_tr:count_tr+ind[i]-1, :] = poses[1][i][:ind[i]-1, :]

        #Get testing data for sequence i
        if seperate_images:
            x_te[0][count_te:count_te+(ind_total[i]-ind[i]),
                :, :, :] = images[i][(ind[i]-1):-1, :, :, :]

            x_te[1][count_te:count_te+(ind_total[i]-ind[i]),
                :, :, :] = images[i][ind[i]:, :, :, :]
        else:
            x_te[count_te:count_te+(ind_total[i]-ind[i]),
                :, :, :image_channels] = images[i][(ind[i]-1):-1, :, :, :]

            x_te[count_te:count_te+(ind_total[i]-ind[i]),
                :, :, image_channels:] = images[i][ind[i]:, :, :, :]

        y_te[0][count_te:count_te+(ind_total[i]-ind[i]), :] = poses[0][i][ind[i]-1:, :]
        y_te[1][count_te:count_te+(ind_total[i]-ind[i]), :] = poses[1][i][ind[i]-1:, :]

        count_tr += ind[i]-1
        count_te += (ind_total[i]-ind[i])

    return x_tr, y_tr, x_te, y_te

def test_datatool():
    '''
    Test function for datatool
    uses sequence 1 (the smallest sequence)
    '''
    get_training_data([1])
    get_training_data([1],seperate_images=False)
#test_datatool()