from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution1D, MaxPooling2D
from keras.layers.normalization import BatchNormalization


def alexNet():
    model = Sequential()
    model.add(Convolution1D(64, 3, 11, 11))
    model.add(BatchNormalization((64, 226, 226)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(poolsize=(3, 3)))

    model.add(Convolution1D(128, 64, 7, 7))
    model.add(BatchNormalization((128, 115, 115)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(poolsize=(3, 3)))

    model.add(Convolution1D(192, 128, 3, 3))
    model.add(BatchNormalization((128, 112, 112)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(poolsize=(3, 3)))

    model.add(Convolution1D(256, 192, 3, 3))
    model.add(BatchNormalization((128, 108, 108)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(poolsize=(3, 3)))

    model.add(Flatten())
    model.add(Dense(512, 4096, init='normal'))
    model.add(BatchNormalization(4096))
    model.add(Activation('relu'))
    model.add(Dense(512, 4096, init='normal'))
    model.add(BatchNormalization(4096))
    model.add(Activation('relu'))
    model.add(Dense(512, 1000, init='normal'))
    model.add(BatchNormalization(1000))
    model.add(Activation('sigmoid'))

    return model, 'sequential'
