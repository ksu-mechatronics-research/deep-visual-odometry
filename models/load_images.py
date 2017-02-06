from os import loaddir
import pykitti
import numpy as np
from scipy.misc import imread

for i in range(list):
    directory = "/Documents/dataset/processed_imgs"
    os.makedirs(directory+"/sequence"+str(i))
    print ("running through sequence: "+str(i))
    for j in range(images[i]):
        imsave(directory+str(j)+".png", images[i][j])