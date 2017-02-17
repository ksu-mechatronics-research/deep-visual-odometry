# The Model of DeepVO
from keras.layers import Input, Merge
from keras.layers.core import Dense, Dropout, Activation, Flatten, Lambda
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.layers.advanced_activations import PReLU
from keras import backend as K #enable tensorflow functions


#AlexNet with batch normalization in Keras
#input image is 128x128

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
    input_img = Input(shape=(128, 128, 6), name='input_img')
    x = Convolution2D(96, 11, 11, border_mode='same')(input_img)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(11, 11), strides=(5, 5), border_mode='same')(x)

    x = Convolution2D(384, 3, 3, border_mode='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=(3, 3), border_mode='same')(x)

    x = Flatten()(x)

    x = Dense(4096, init='normal')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Dense(4096, init='normal')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)


    # Delta Translation output
    translation_proc = Dense(3, init='normal')(x)
    vector_translation = Activation(PReLU(), name='translation')(translation_proc)
    
    # Delta rotation in quaternion form
    rotation_proc = Dense(64, activation='relu')(x)
    rotation_proc = Dense(64, activation='relu')(rotation_proc)
    rotation_proc = Dense(64, activation='relu')(rotation_proc)
    rotation_proc = Dense(4, activation='tanh')(rotation_proc)
    quaternion_rotation = Lambda(normalize_quaternion, name='rotation')(rotation_proc)

    model = Model(input=input_img, output=[vector_translation, quaternion_rotation])

    return model

def normalize_quaternion(x):
    "use tensorflow normalize function on this layer to ensure valid quaternion rotation"
    x = K.l2_normalize(x, axis=1)
    return x


def train_model(model, Xtr, Ytr, Xte, Yte, save_path=None):
    "Note: y should be [[translation],[quat rotation]]"
    
    model.compile(loss='mean_squared_error', optimizer='adam',
                                        metrics=['mean_absolute_error'])

    history = model.fit(Xtr, Ytr, validation_split=0.2, batch_size=8, nb_epoch=30, verbose=1)

    score = model.evaluate(Xte, Yte, verbose=1)

    if (save_path != None):
        model.save(save_path)

    return score, history
