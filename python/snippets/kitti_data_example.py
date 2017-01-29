#############################################################
#
# This is an example script demonstrating how to access data
#    from the kitti dataset.
#
# look at https://github.com/utiasSTARS/pykitti/blob/master/pykitti/odometry.py
#    as a reference for pikitti.
#
##############################################################
import pykitti

# Path to dataset: folder containing sequences, poses
basedir = '/home/sexy/Documents/dataset'
sequence = '03'

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

