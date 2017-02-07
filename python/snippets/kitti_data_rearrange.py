#############################################################
#
# This is an example script demonstrating how to access data
#    from the kitti dataset.
#
# look at https://github.com/utiasSTARS/pykitti/blob/master/pykitti/odometry.py
#    as a reference for pikitti.
#
##############################################################
import os
import pykitti
import numpy
from scipy.misc import imread, imresize, imsave
from shutil import copyfile

# crop function
def crop(image):
    if image.shape[0] > image.shape[1]:
        square_dim = int((image.shape[0]-image.shape[1])/2)
        return image[square_dim:-square_dim,:]
    else:
        square_dim = int((image.shape[1]-image.shape[0])/2)
        return image[:, square_dim:-square_dim]

# Path to dataset: folder containing sequences, poses
basedir = '/home/sexy/Documents/dataset/'
# Target path
targetdir = '/home/sexy/Documents/trial1'

#Go through each directory:
images = [[] for i in range(11)]
for q in range(11):
    i = q
    sequence = str(i).zfill(2)
    print ("running through sequence: "+str(sequence))
    directory = basedir+"sequences/"+sequence+"/image_2/"
    directoryW = "/home/sexy/Documents/dataset/processed_imgs_128/"
    #os.makedirs(directoryW+"/sequence"+str(i))
    count = 0
    #print (type(os.listdir(directory)))
    a = os.listdir(directory)
    a.sort()
    for j in a:
        print(directory+j)
        image = imread(directory+j)
        #images[i].append(imresize(crop(image), (256,256)))
        print(directoryW+"sequence"+str(i)+"/"+str(count).zfill(5)+".png")
        imsave(directoryW+"sequence"+str(i)+"/"+str(count).zfill(5)+".png", imresize(crop(image), (128,128)))
        count += 1
    #copyfile(basedir+'/'+sequence+'/times.txt', dst)
print ("done with reading")
'''
for i in range(len(images)):
    directoryW = "/Documents/dataset/processed_imgs"
    os.makedirs(directoryW+"/sequence"+str(i))
    print ("running through sequence: "+str(i))
    for j in range(len(images[i])):
        imsave(directory+str(j)+".png", images[i][j])


#Make paths:
paths = [
"images/test",
"images/train",
"odometry/test",
"odometry/train"
]

#Add paths for folders 0-11
for i in range(11):
    paths.append("images/"+str(i).zfill(2))
    paths.append("odometry/"+str(i).zfill(2))

# print(paths)

# Make directories
if not os.path.exists(targetdir):
    for path in paths:
         os.makedirs(targetdir+'/'+path)

paths = []


sequence = '00'

# The range argument is optional - default is None, which loads the whole dataset
data = pykitti.odometry(basedir, sequence)

# Data is loaded only if requested
data.load_timestamps()
data.load_poses()
data.load_rgb()

print('timestamps type:',)
print(type(data.timestamps))
#data.timestamps is a list

print("image data type:",)
print(type(data.rgb))

cam2_image = data.rgb[0].left
print("image shape:",)
print(cam2_image.shape)
print('image type:',)
print(type(cam2_image))

#odometry data is centered on the left camera
print("odometry data shape:", data.T_w_cam0.shape)
print("odometry data example:")
print(data.T_w_cam0[0])
'''
