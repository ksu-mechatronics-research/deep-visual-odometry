# The Model of DeepVO using relative orientation 
from keras.layers import Input, Merge
from keras.layers.core import Dense, Dropout, Activation, Flatten, Lambda
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.models import Model, Sequential
from keras.layers.advanced_activations import PReLU
from keras import backend as K #enable tensorflow functions

def add_convolution(model, filters, kernalSize, activation, poolSize, poolStride):
    '''
    returns a single convolution layer equiped with BatchNormalization, an activation, and MaxPooling2D
    This only contains Convolution2D
    args:
        filters:
            the number of filters in the Convolution2D
        kernalSize:
            assuming a square kernal with a side length of kernalSize
        activation: 
            a string with the activation type, use 'prelu' for prelus
        poolSize:
            assuming a square kernal with a side length of poolSize for MaxPooling2D
        poolStride:
            assuming a square stride with a distance of poolSize for MaxPooling2D
    '''
    model = Convolution2D(filters, kernalSize, kernalSize, border_mode='same')(model)
    model = BatchNormalization()(model)
    if activation == "prelu" :
        model = PReLU()(model)
    else:
        model = Activation(activation)(model)
    model = MaxPooling2D(pool_size=(poolSize, poolSize), strides=(poolStride, poolStride), border_mode='same')(model)
    return model

def add_dense(model, denseSize, activation):
    '''
    returns a single dense layer equiped with BatchNormalization, and an activation
    args:
        denseSize:
            the number of neurons for the fully connected dense layer
        activation:
            a string with the activation type, use 'prelu' for prelus
    '''
    model = Dense(denseSize, init='normal')(model)
    model = BatchNormalization()(model)
    if activation == "prelu":
        model = PReLU()(model)
    else:
        model = Activation(activation)(model)
    return model

def normalize_quaternion(x):
    "Use tensorflow normalize function on this layer to ensure valid quaternion rotation"
    x = K.l2_normalize(x, axis=1)
    return x

def add_quaternion_dense(model):
    # Delta Translation output
    translation_proc = Dense(3, init='normal')(model)
    vector_translation = Activation(PReLU(), name='translation')(translation_proc)

    # Delta rotation in quaternion form
    rotation_proc = Dense(64, activation='relu')(model)
    rotation_proc = Dense(64, activation='relu')(rotation_proc)
    rotation_proc = Dense(64, activation='relu')(rotation_proc)
    rotation_proc = Dense(4, activation='tanh')(rotation_proc)
    quaternion_rotation = Lambda(normalize_quaternion, name='rotation')(rotation_proc)

    return vector_translation, quaternion_rotation

def create_model(conv_params, dense_params, quaternion=False):
    '''
    return a built model based on the parmeters in list format
    args:
        conv_params:
            a list of parameters for the convolution layers, more info in the add_convolution function
            0: filters
            1: kernalSize
            2: activation
            3: poolSize
            4: poolStride

        dense_params:
            a list of parameters for the dense layers, more info in the add_dense function
            0: denseSize
            1: activation
    '''
    input_shape = Input(shape=(128, 128, 6), name='input_img')
    model = input_shape

    for i, param in enumerate(conv_params):
        model = add_convolution(model, param[0], param[1], param[2], param[3], param[4])
    model = Flatten()(model)
    for i, param in enumerate(dense_params):
        model = add_dense(model, param[0], param[1])
        # check if this is the last iteration
        if i == len(dense_params)-1:
            if quaternion == False:
                model = add_dense(model, 3, 'linear')
    
    if quaternion:
        trans, quat = add_quaternion_dense(model)
        return Model(input=input_shape, output=[trans, quat])
    return Model(input=input_shape, output=model)

def train_model(model, Xtr, Ytr, save_path=None):
    '''
    trains the model and returns the history of the training process
    args:
        model:
            the model that is being trained on
        Xtr:
            the input training data
        Ytr:
            the ground truth that it is supervised with
        save_path:
            path to save the model, default is None and will not save the model
    '''
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_absolute_error'])

    history = model.fit(Xtr, Ytr, validation_split=0.2, batch_size=1, nb_epoch=30, verbose=1)

    if save_path != None:
        model.save(save_path)

    return history







