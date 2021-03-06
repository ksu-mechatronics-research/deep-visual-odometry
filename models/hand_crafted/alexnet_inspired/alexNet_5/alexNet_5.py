# The Model of DeepVO
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten, Reshape
from keras.layers.advanced_activations import PReLU
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.recurrent import LSTM

#AlexNet with batch normalization in Keras
#input image is 224x224

def create_model():
    model = Sequential()
    
    model.add(Convolution2D(96, 11, 11, border_mode='same', input_shape=(128, 128, 6)))
    model.add(BatchNormalization())
    model.add(PReLU())
    model.add(MaxPooling2D(pool_size=(11, 11), strides=(1, 1), border_mode='same'))

    #model.add(Convolution2D(256, 5, 5, border_mode='same'))
    #model.add(BatchNormalization())
    #model.add(PReLU())
    #model.add(MaxPooling2D(pool_size=(5, 5), strides=(1, 1), border_mode='same'))

    #model.add(Convolution2D(384, 3, 3, border_mode='same'))
    #model.add(BatchNormalization())
    #model.add(PReLU())
    #model.add(MaxPooling2D(pool_size=(3, 3), strides=(1, 1), border_mode='same'))

    model.add(Convolution2D(384, 3, 3, border_mode='same'))
    model.add(BatchNormalization())
    model.add(PReLU())
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(1, 1), border_mode='same'))
    
    model.add(Flatten())
    print (model.output_shape)
    
    model.add(Reshape((384, 16384), input_shape=(6291456,)))
    
    # model shape == (samples, pooled_rows, pooled_cols, channels)
    # model shape == (samples, 128, 128, 384)
    
    model.add(LSTM(64, consume_less='gpu'))
    model.add(BatchNormalization())
    model.add(PReLU())
    
    print (model.output_shape)

    model.add(Dense(4096, init='normal'))
    model.add(BatchNormalization())
    model.add(PReLU())

    model.add(Dense(4096, init='normal'))
    model.add(BatchNormalization())
    model.add(PReLU())

    model.add(Dense(1024, init='normal'))
    model.add(BatchNormalization())
    model.add(Activation('tanh'))

    model.add(Dense(3, init='normal'))
    model.add(BatchNormalization())
    model.add(Activation('linear'))

    return model

def run_model(model, Xtr, Ytr, Xte, Yte, save_path=None):
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_absolute_error'])

    history = model.fit(Xtr, Ytr, batch_size=8, nb_epoch=30, verbose=1).history

    score = model.evaluate(Xte, Yte, verbose=1)

    if (save_path != None):
        model.save(save_path)

    return score, history













