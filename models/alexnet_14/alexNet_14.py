# The Model of DeepVO
from keras.layers.core import Dense, Dropout, Activation, Flatten, Lambda
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras import backend as K #enable tensorflow functions


#AlexNet with batch normalization in Keras
#input image is 128x128

def create_model():
    """
    This model is designed to take in multiple inputs and give multiple outputs. 
    Here is what the network was designed for: 
    
    Inputs:
    two 128x128 RGB images stacked (RGBRGB)

    Outputs:
    Translation between two images
    Rotation between images in quaternion form
    """
    main_input = Convolution2D(96, 11, 11, border_mode='same', input_shape=(128, 128, 6), name='main_input')
    x = BatchNormalization()(main_input)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(11, 11), strides=(1, 1), border_mode='same')(x)

    x = Convolution2D(384, 3, 3, border_mode='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), border_mode='same')(x)

    x = Flatten()(x)

    x = Dense(4096, init='normal')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Dense(4096, init='normal')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # Delta rotation in quaternion form
    quaternion_rotation =  Dense(4, activation='tanh', name='quaternion_rotation')(x)
    quaternion_rotation = Lambda(normalize_quaternion)(quaternion_rotation)

    # Delta Translation output
    translation = Dense(3, activation='linear', name='translation')(x)

    model = Model(input=main_input, output=[translation, quaternion_rotation])
    
    return model

def normalize_quaternion(x):
    "use tensorflow normalize function on this layer to ensure valid quaternion rotation"
    x = K.l2_normalize(x, dim=1)
    return x


def run_model(model, Xtr, Ytr, Xte, Yte, save_path=None):
    "Note: y should be [[translation],[quat rotation]]
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_absolute_error'])

    history = model.fit(Xtr, Ytr, batch_size=8, nb_epoch=30, verbose=1).history

    score = model.evaluate(Xte, Yte, verbose=1)

    if (save_path != None):
        model.save(save_path)

    return score, history













