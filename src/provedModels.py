from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.layers.normalization import BatchNormalization
from keras.layers import  ZeroPadding1D
from keras import optimizers

from evaluation import auc_roc


def alexNet(x_train):   # Adapted from https://gist.github.com/JBed/c2fb3ce8ed299f197eff
    model = Sequential()
    model.add(Convolution1D(64, 3, input_shape=(x_train.shape[1], 1)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling1D(pool_size=3))

    model.add(Convolution1D(128, 7))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling1D(pool_size=3))

    model.add(Convolution1D(192, 3))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling1D(pool_size=3))

    model.add(Convolution1D(256, 3))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling1D(pool_size=3))

    model.add(Flatten())
    model.add(Dense(512, init='normal'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dense(512, init='normal'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dense(1, init='normal'))
    model.add(BatchNormalization())
    model.add(Activation('sigmoid'))

    opt = optimizers.Adam(learning_rate=10e-5, beta_1=0.9,
                          beta_2=0.999, amsgrad=False)
    model.compile(loss='binary_crossentropy', optimizer=opt,
                  metrics=['accuracy', auc_roc])
    return model, 'sequential'


def vgg(x_train):   # Adapted from https://gist.github.com/baraldilorenzo/07d7802847aaad0a35d3
    model = Sequential()
    model.add(ZeroPadding1D(1, input_shape=(x_train.shape[1], 1)))
    model.add(Convolution1D(64, 3, 3, activation='relu'))
    model.add(ZeroPadding1D(1))
    model.add(Convolution1D(64, 3, 3, activation='relu'))
    model.add(MaxPooling1D(2, strides=2))

    model.add(ZeroPadding1D(1))
    model.add(Convolution1D(128, 3, 3, activation='relu'))
    model.add(ZeroPadding1D(1))
    model.add(Convolution1D(128, 3, 3, activation='relu'))
    model.add(MaxPooling1D(2, strides=2))

    model.add(ZeroPadding1D(1))
    model.add(Convolution1D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding1D(1))
    model.add(Convolution1D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding1D(1))
    model.add(Convolution1D(256, 3, 3, activation='relu'))
    model.add(MaxPooling1D(2, strides=2))

    model.add(ZeroPadding1D(1))
    model.add(Convolution1D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding1D(1))
    model.add(Convolution1D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding1D(1))
    model.add(Convolution1D(512, 3, 3, activation='relu'))
    model.add(MaxPooling1D(2, strides=2))

    model.add(ZeroPadding1D(1))
    model.add(Convolution1D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding1D(1))
    model.add(Convolution1D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding1D(1))
    model.add(Convolution1D(512, 3, 3, activation='relu'))
    model.add(MaxPooling1D(2, strides=2))

    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    return model, 'sequential'
