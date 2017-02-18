from keras.layers import Input
from keras.layers.core import Flatten, Dense, Dropout, Lambda
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras import backend as K
from keras.optimizers import Adam

def VGG_16():
    input_img = Input(shape=(128, 128, 6), name='input_img')
    x = ZeroPadding2D((1,1),input_shape=(128,128,6))(input_img)
    x = Convolution2D(64, 3, 3, activation='relu')(x)
    x = BatchNormalization()(x)
    x = ZeroPadding2D((1,1))(x)
    x = Convolution2D(64, 3, 3, activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2,2), strides=(2,2))(x)

    x = ZeroPadding2D((1,1))(x)
    x = Convolution2D(128, 3, 3, activation='relu')(x)
    x = BatchNormalization()(x)
    x = ZeroPadding2D((1,1))(x)
    x = Convolution2D(128, 3, 3, activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2,2), strides=(2,2))(x)

    x = ZeroPadding2D((1,1))(x)
    x = Convolution2D(256, 3, 3, activation='relu')(x)
    x = BatchNormalization()(x)
    x = ZeroPadding2D((1,1))(x)
    x = Convolution2D(256, 3, 3, activation='relu')(x)
    x = BatchNormalization()(x)
    x = ZeroPadding2D((1,1))(x)
    x = Convolution2D(256, 3, 3, activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2,2), strides=(2,2))(x)

    x = ZeroPadding2D((1,1))(x)
    x = Convolution2D(512, 3, 3, activation='relu')(x)
    x = BatchNormalization()(x)
    x = ZeroPadding2D((1,1))(x)
    x = Convolution2D(512, 3, 3, activation='relu')(x)
    x = BatchNormalization()(x)
    x = ZeroPadding2D((1,1))(x)
    x = Convolution2D(512, 3, 3, activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2,2), strides=(2,2))(x)

    x = ZeroPadding2D((1,1))(x)
    x = Convolution2D(512, 3, 3, activation='relu')(x)
    x = BatchNormalization()(x)
    x = ZeroPadding2D((1,1))(x)
    x = Convolution2D(512, 3, 3, activation='relu')(x)
    x = BatchNormalization()(x)
    x = ZeroPadding2D((1,1))(x)
    x = Convolution2D(512, 3, 3, activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2,2), strides=(2,2))(x)

    x = Flatten()(x)
    x = Dense(4096, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(4096, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(1024, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)

    # Delta Translation output
    vector_translation = Dense(3, init='normal', activation='linear', name='translation')(x)
    
    # Delta rotation in quaternion form
    rotation_proc = Dense(4, init='normal', activation='tanh')(x)
    quaternion_rotation = Lambda(normalize_quaternion, name='rotation')(rotation_proc)
    
    model = Model(input=input_img, output=[vector_translation, quaternion_rotation])
    
    return model

def normalize_quaternion(x):
    "use tensorflow normalize function on this layer to ensure valid quaternion rotation"
    x = K.l2_normalize(x, axis=1)
    return x

def train_model(model, Xtr, Ytr, Xte, Yte, save_path=None):
    "Note: y should be [[translation],[quat rotation]]"
    
    adam = Adam(lr=0.017, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(loss='mean_squared_error', optimizer=adam, metrics=['mean_absolute_error'])

    history = model.fit(Xtr, Ytr, validation_split=0.2, batch_size=32, nb_epoch=10, verbose=1)

    score = model.evaluate(Xte, Yte, verbose=1)

    if save_path:
        model.save(save_path)

    return score, history


#if __name__ == "__main__":
#    im = cv2.resize(cv2.imread('cat.jpg'), (224, 224)).astype(np.float32)
#    im[:,:,0] -= 103.939
#    im[:,:,1] -= 116.779
#    im[:,:,2] -= 123.68
#    im = im.transpose((2,0,1))
#    im = np.expand_dims(im, axis=0)

    # Test pretrained model
#    model = VGG_16('vgg16_weights.h5')
#    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
#    model.compile(optimizer=sgd, loss='categorical_crossentropy')
#    out = model.predict(im)

#print np.argmax(out)
