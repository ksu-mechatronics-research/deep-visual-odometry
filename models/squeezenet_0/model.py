# The Model of DeepVO
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

    input_img = Input(shape=(128, 128, 6), name='input_img')
    
    x = Convolution2D(64, 3, 3, subsample=(2, 2), border_mode='valid', name='conv1')(input_img)
    x = Activation('relu', name='relu_conv1')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool1')(x)

    x = fire_module(x, fire_id=2, squeeze=16, expand=64)
    x = fire_module(x, fire_id=3, squeeze=16, expand=64)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool3')(x)

    x = fire_module(x, fire_id=4, squeeze=32, expand=128)
    x = fire_module(x, fire_id=5, squeeze=32, expand=128)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool5')(x)

    x = fire_module(x, fire_id=6, squeeze=48, expand=192)
    x = fire_module(x, fire_id=7, squeeze=48, expand=192)
    x = fire_module(x, fire_id=8, squeeze=64, expand=256)
    x = fire_module(x, fire_id=9, squeeze=64, expand=25)
    x = Dropout(0.5, name='drop9')(x)
    x = BatchNormalization()(x)
    x = Convolution2D(4096, 1, 1, border_mode='valid', name='conv10')(x)
    x = Activation('relu', name='relu_conv10')(x)
    x = GlobalAveragePooling2D()(x)
    
    #x = Flatten()(x)

    x = Dense(4096, init='normal')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
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
    rotation_proc = Dense(4, activation='tanh')(x)
    quaternion_rotation = Lambda(normalize_quaternion, name='rotation')(rotation_proc)

    model = Model(input=input_img, output=[vector_translation, quaternion_rotation])

    return model

def fire_module(x, fire_id, squeeze=16, expand=64):
    s_id = 'fire' + str(fire_id) + '/'

    x = Convolution2D(squeeze, 1, 1, border_mode='valid', name=s_id + sq1x1)(x)
    x = Activation('relu', name=s_id + relu + sq1x1)(x)

    left = Convolution2D(expand, 1, 1, border_mode='valid', name=s_id + exp1x1)(x)
    left = Activation('relu', name=s_id + relu + exp1x1)(left)

    right = Convolution2D(expand, 3, 3, border_mode='same', name=s_id + exp3x3)(x)
    right = Activation('relu', name=s_id + relu + exp3x3)(right)

    x = merge([left, right], mode='concat', concat_axis=3, name=s_id + 'concat')
    return x

def normalize_quaternion(x):
    "use tensorflow normalize function on this layer to ensure valid quaternion rotation"
    x = K.l2_normalize(x, axis=1)
    return x


def train_model(model, Xtr, Ytr, Xte, Yte, save_path=None):
    "Note: y should be [[translation],[quat rotation]]"
    
    model.compile(loss='mean_squared_error', optimizer=Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0),
                                        metrics=['mean_absolute_error'])

    history = model.fit(Xtr, Ytr, validation_split=0.2, batch_size=64, nb_epoch=30, verbose=1)

    score = model.evaluate(Xte, Yte, verbose=1)

    if (save_path != None):
        model.save(save_path)

    return score, history
