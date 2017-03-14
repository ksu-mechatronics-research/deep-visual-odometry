'''
This is a script to take a rotation (quaternion) model and generate
    jsons containing path info
'''
import os
import json
import numpy as np
import quaternion
from datatool import get_eval_data, load_poses
from keras.models import load_model
import matplotlib.pyplot as plt

def quat_dump_paths(model_file_path=''):
    '''
    Takes in model path and evaluates it.
    '''
    if not model_file_path:
        raise ValueError('no model path specified')

    # load model from given path
    model = load_model(model_file_path)

    # get evaluation data
    eval_data = get_eval_data(False)
    rotations = quaternion.as_quat_array(load_poses()[1])

    # evaluate performance
    # Note: predictions contains 11 lists,
    #     each containing 2 lists: translation and quaternion rotation
    predictions = []
    for i in range(11):
        predictions.append(model.predict(eval_data[i], batch_size=8, verbose=1))
        predictions[i][0] = (predictions[i][0]).astype(np.float)

    global_rotations =rotations
    pred_global_delta_trans = []
    pred_position = []
    for i in range(11):
        #Undo rotation on translation deltas
        pred_global_delta_trans.append(np.ndarray((len(predictions[i][0]+1), 3)))
        for j in range(len(predictions[i][0])):
            pred_global_delta_trans[i][j] = (global_rotations[i][j].conj()*quaternion.quaternion(0.,*predictions[i][0][j])*global_rotations[i][j]).vec

        # get final global translation data
        pred_position.append(np.ndarray((len(predictions[i][0]+1), 3)))
        for j in range(len(predictions[i][0]+1)):
            pred_position[i][j, :] = np.sum(pred_global_delta_trans[i][:j], axis=0)
#        plt.subplot(3,4,i+1)
#        plt.axis('equal')
#        plt.plot(pred_position[i][:,0], pred_position[i][:,2])
#    plt.show()

    # convert numpy array to json serializable lists:
    for i, pos in enumerate(pred_position):
        pred_position[i] = pos.tolist()

    # save position datas
    with open(os.path.join(model_file_path[:-3]+"_path.json"), 'w') as f:
        json.dump(pred_position, f, indent=4)
