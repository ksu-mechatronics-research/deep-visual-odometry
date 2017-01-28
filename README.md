# deep-visual-odometry
[![license](https://img.shields.io/github/license/mashape/apistatus.svg?maxAge=2592000)](https://github.com/fchollet/keras/blob/master/LICENSE)
## What is this?

Visual odometry is the use of visual sensors to estimate change in position over time.

The state of the art is to accomplish this with techniques such as SLAM, but these techniques are resource intensive and run slowly on limited hardware platforms such as Odroid or Raspberry Pi.

With this project, we seek to create a neural network capable of reliably and efficiently performing visual odometry.


## Environment:
We are using python3 with keras and tensorflow.

### on Ubuntu 16.04:
install tensorflow (with or without gpu)

    pip3 install tensorflow-gpu
    #or
    pip3 install tensorflow

install keras:

    pip3 install hdf5
    pip3 install pyyaml
    pip3 install numpy
    pip3 install scipy
    pip3 install keras

install kitti dataset tool:

    pip install pykitti
