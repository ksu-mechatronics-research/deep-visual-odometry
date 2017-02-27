# The relative translation and rotation model labeled siamese_quat
from keras.layers import Input, merge
from keras.layers.core import Dense, Dropout, Activation, Flatten, Lambda
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.layers.advanced_activations import PReLU
from keras import backend as K #enable tensorflow functions
from keras.layers.pooling import GlobalAveragePooling2D
from keras.optimizers import Adam


#AlexNet with batch normalization in Keras
#input image is 128x128

sq1x1 = "squeeze1x1"
exp1x1 = "expand1x1"
exp3x3 = "expand3x3"
relu = "relu_"
    
def create_model():
    """
    This model is designed to take in images and give multiple outputs. 
    Here is what the network was designed for: 

    Inputs:
    128x128X6 RGB images stacked (RGBRGB)

    Outputs:
    Translation between two images
    Rotation between images in quaternion form
    """
    #input images:
    input_initial = Input(shape=(128, 128, 3), name='input_img_initial')
    input_initial = Input(shape=(128, 128, 3), name='input_img_final')


    #Siamese part (shared weights):
    x = Convolution2D(64, 3, 3, subsample=(2, 2), border_mode='valid', name='conv1')
    x = Activation('relu', name='relu_conv1')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool1')(x)

    x = fire_module(x, fire_id=2, squeeze=16, expand=64)
    x = fire_module(x, fire_id=3, squeeze=16, expand=64)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool3')(x)

    x = fire_module(x, fire_id=4, squeeze=32, expand=128)
    x = fire_module(x, fire_id=5, squeeze=32, expand=128)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool5')(x)


    # Delta Translation output
    vector_translation = Dense(3, init='normal', activation=PReLU(), name='translation')(x)

    # Delta rotation in quaternion form
    rotation_proc = Dense(4, init='normal', activation=PReLU())(x)
    quaternion_rotation = Lambda(normalize_quaternion, name='rotation')(rotation_proc)

    model = Model(input=input_img, output=[vector_translation, quaternion_rotation])

    return model

def fire_module(x, fire_id, squeeze=16, expand=64):
    s_id = 'fire' + str(fire_id) + '/'

    x = Convolution2D(squeeze, 1, 1, border_mode='valid', name=s_id + sq1x1)(x)
    x = Activation('relu', name=s_id + relu + sq1x1)(x)
    x = BatchNormalization()(x)

    left = Convolution2D(expand, 1, 1, border_mode='valid', name=s_id + exp1x1)(x)
    left = Activation('relu', name=s_id + relu + exp1x1)(left)
    left = BatchNormalization()(left)

    right = Convolution2D(expand, 3, 3, border_mode='same', name=s_id + exp3x3)(x)
    right = Activation('relu', name=s_id + relu + exp3x3)(right)
    right = BatchNormalization()(right)

    x = merge([left, right], mode='concat', concat_axis=3, name=s_id + 'concat')
    return x

def normalize_quaternion(x):
    "use tensorflow normalize function on this layer to ensure valid quaternion rotation"
    x = K.l2_normalize(x, axis=1)
    return x
