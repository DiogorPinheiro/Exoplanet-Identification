from keras.models import Sequential, Model
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.layers.normalization import BatchNormalization
from keras.layers import ZeroPadding1D, Input
from keras import optimizers

# Adaptation for time-series data of some well know NN models


def alexNet(x_train):
    # Adapted from https://gist.github.com/JBed/c2fb3ce8ed299f197eff

    model_input = Input(shape=(x_train.shape[1], 1))

    model = Convolution1D(filters=64, kernel_size=3)(model_input)
    model = BatchNormalization()(model)
    model = Activation('relu')(model)
    model = MaxPooling1D(pool_size=3)(model)

    model = Convolution1D(filters=128, kernel_size=7)(model)
    model = BatchNormalization()(model)
    model = Activation('relu')(model)
    model = MaxPooling1D(pool_size=3)(model)

    model = Convolution1D(filters=192, kernel_size=3)(model)
    model = BatchNormalization()(model)
    model = Activation('relu')(model)
    model = MaxPooling1D(pool_size=3)(model)

    model = Convolution1D(filters=256, kernel_size=3)(model)
    model = BatchNormalization()(model)
    model = Activation('relu')(model)
    model = MaxPooling1D(pool_size=3)(model)

    model = Flatten()(model)
    model = Dense(512, init='normal')(model)
    model = BatchNormalization()(model)
    model = Activation('relu')(model)
    model = Dense(512, init='normal')(model)
    model = BatchNormalization()(model)
    model = Activation('relu')(model)

    model = Dense(1, init='normal')(model)
    model = BatchNormalization()(model)
    model = Activation('sigmoid')(model)

    model = Model(inputs=model_input, outputs=model)

    opt = optimizers.Adam(learning_rate=10e-5, beta_1=0.9,
                          beta_2=0.999, amsgrad=False)
    model.compile(loss=customLoss, optimizer=opt,
                  metrics=['accuracy', auc_roc])
    return model, 'functional'


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
