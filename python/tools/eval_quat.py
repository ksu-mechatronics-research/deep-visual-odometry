'''
This is a script to take a rotation (quaternion) model and generate
    jsons containing path information.
'''
import os
import json
from datatool import get_training_data, load_poses
from keras.models import load_model

def rot_generate_paths(model_file_path=''):
    '''
    Takes in model path and evaluates it.
    '''
    if not model_file_path:
        raise ValueError('no model path specified')

    # load model from given path
    model = load_model(model_file_path)

    # get dataset data
    x_tr, y_tr = get_training_data()

    #get initial rotations:
    initial_rotations = load_poses()[1][:][0]

    # evaluate performance
    # Note: predictions contains 11 lists,
    #     each containing 2 lists: translation and quaternion rotation
    predictions = []
    for i in range(11):
        predictions.append(model.predict(x_tr[i], batch_size=64, verbose=1))
        
    