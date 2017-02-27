# The Model of DeepVO using relative orientation 
from keras.layers import Input, Merge
from keras.layers.core import Dense, Dropout, Activation, Flatten, Lambda
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.layers.advanced_activations import PReLU
from keras import backend as K #enable tensorflow functions

def create_model():    
    """
    Input:
    sequential images of 128x128x3 + 128x128x3 = 128x128x6
    in RGBRGB format

    Output:
    three values for relative translations delta(x,y,z)
    """

    input_img = Input(shape=(128, 128, 6), name='input_img')
    x = Convolution2D(96, 11, 11, border_mode='same')(input_img)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=(3, 3), border_mode='same')(x)

    x = Convolution2D(256, 5, 5, border_mode='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=(3, 3), border_mode='same')(x)

    x = Convolution2D(384, 3, 3, border_mode='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=(3, 3), border_mode='same')(x)

    x = Convolution2D(384, 3, 3, border_mode='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=(3, 3), border_mode='same')(x)

    x = Flatten()(x)

    x = Dense(4096, init='normal')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Dense(2048, init='normal')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Dense(1024, init='normal')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # Delta Translation output
    translation_proc = Dense(3, init='normal')(x)
    vector_translation = Activation(PReLU(), name='translation')(translation_proc)

    model = Model(input=input_img, output=vector_translation)

    return model

def train_model(model, Xtr, Ytr, save_path=None):
    """
    
    returns the history only(loss and validation)
    """
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_absolute_error'])

    history = model.fit(Xtr, Ytr, validation_split=0.2, batch_size=8, nb_epoch=30, verbose=1)

    if (save_path != None):
        model.save(save_path)

    return history







