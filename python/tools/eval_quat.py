'''
This is a script to take a rotation (quaternion) model and generate
    jsons containing path information.
'''
import os
import json
import numpy as np
import quaternion
from datatool import get_eval_data
from keras.models import load_model
import matplotlib.pyplot as plt

def quat_generate_paths(model_file_path=''):
    '''
    Takes in model path and evaluates it.
    '''
    if not model_file_path:
        raise ValueError('no model path specified')

    # load model from given path
    model = load_model(model_file_path)

    # get evaluation data
    eval_data, initial_rotations = get_eval_data()
    initial_rotations = quaternion.as_quat_array(initial_rotations)

    # evaluate performance
    # Note: predictions contains 11 lists,
    #     each containing 2 lists: translation and quaternion rotation
    predictions = []
    for i in range(11):
        predictions.append(model.predict(eval_data[i], batch_size=8, verbose=1))
        predictions[i][0] = (predictions[i][0]).astype(np.float)
        predictions[i][1] = (predictions[i][1]).astype(np.float)

    pred_global_rotations = []
    pred_global_delta_trans = []
    pred_position = []
    for i in range(11):
        #Undo rotation delta
        predictions[i][1] = quaternion.as_quat_array(predictions[i][1])
        pred_global_rotations.append(np.ndarray(len(predictions[i][1]+1), dtype=quaternion.quaternion))
        pred_global_rotations[i][0] = initial_rotations[i]
        for j in range(len(predictions[i][1])-1):
            pred_global_rotations[i][j+1] = pred_global_rotations[i][j]*predictions[i][1][j]

        #Undo rotation on translation deltas
        pred_global_delta_trans.append(np.ndarray((len(predictions[i][0]+1), 3)))
        for j in range(len(predictions[i][0])):
            pred_global_delta_trans[i][j] = (pred_global_rotations[i][j].conj()*quaternion.quaternion(0.,*predictions[i][0][j])*pred_global_rotations[i][j]).vec

        # get final global translation data
        pred_position.append(np.ndarray((len(predictions[i][0]+1), 3)))
        for j in range(len(predictions[i][0]+1)):
            pred_position[i][j, :] = np.sum(pred_global_delta_trans[i][:j], axis=0)
        plt.subplot(3,4,i+1)
        plt.axis('equal')
        plt.plot(pred_position[i][:,0], pred_position[i][:,2])
    plt.show()
