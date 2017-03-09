'''
python script to test graphing functionality
'''
import os
import sys
PATH = os.getcwd()
sys.path.append(os.path.join(PATH, '..', 'tools'))
from eval_quat import quat_generate_paths


model_file = os.path.join(PATH, '..', '..', 'models', 'quat_model_1', 'train_2.h5')

quat_generate_paths(model_file)
