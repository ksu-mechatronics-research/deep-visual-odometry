'''
python script to test graphing functionality
'''
import os
import sys
PATH = os.getcwd()
sys.path.append(os.path.join(PATH, '..', 'tools'))
from eval_quat import quat_dump_paths

def get_models(_path=''):
    files = os.listdir(_path)
    model_files = []
    for i in files:
        if '.h5' in i:
            model_files.append(i)
    return model_files

path = os.path.join(PATH, '..', '..', 'models', 'quat_models')
quat_model_files = get_models(path)
for i in quat_model_files:
    quat_dump_paths(os.path.join(path, i))
