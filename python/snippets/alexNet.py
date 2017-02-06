# The Model of DeepVO
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization

#AlexNet with batch normalization in Keras
#input image is 224x224

def create_model():
    model = Sequential()
    model.add(Convolution2D(96, 11, 11, border_mode='same', input_shape=(256, 256, 6)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(4, 4)))
    
    model.add(Convolution2D(256, 5, 5, border_mode='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(3, 3)))
    
    model.add(Convolution2D(384, 3, 3, border_mode='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(3, 3)))
    
    model.add(Convolution2D(384, 3, 3, border_mode='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(3, 3)))
    
    model.add(Flatten())
    model.add(Dense(4096, init='normal'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    
    model.add(Dense(4096, init='normal'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    
    model.add(Dense(4096, init='normal'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    
    model.add(Dense(1024, init='normal'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    
    model.add(Dense(3, init='normal'))
    model.add(BatchNormalization())
    model.add(Activation('softmax'))

    return model

def run_model(model, Xtr, Ytr, Xte, Yte):
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
    
    model.fit(Xtr, Ytr, batch_size=8, nb_epoch=60, verbose=1)
    
    score = model.evaluate(Xte, Yte, verbose=1)
    
    return score













    
