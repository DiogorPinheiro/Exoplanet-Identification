from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.model_selection import RandomizedSearchCV
from keras.models import Sequential
from keras.layers import Dense, Input, Flatten
from keras.models import Model
import keras


def knn():
    model = KNeighborsClassifier(n_neighbors=3)

    return model


def svmachine():
    svr = svm.SVC(probability=False)

    parameters = {'kernel': ('linear', 'rbf'), 'C': [1, 5]}
    model = RandomizedSearchCV(svr, parameters, n_jobs=4, verbose=5)

    return model


def feedForwardNN(x_train_global, x_train_local):
    inputLayer_local = Input(shape=(x_train_local.shape[1], 1))
    inputLayer_global = Input(shape=(x_train_global.shape[1], 1))

    conv_local = Dense(16, kernel_size=5, strides=1, padding='same', dilation_rate=1,
                       activation='relu', kernel_initializer='he_normal')
    conv_global = Dense(16,  kernel_size=5, strides=1, padding='same', dilation_rate=1,
                        activation='relu', kernel_initializer='he_normal')

    model1 = conv_global(inputLayer_global)
    model1 = Dense(16,  kernel_size=5, strides=1, padding='same',
                   dilation_rate=1, activation='relu')(model1)
    model1 = Flatten()(model1)

    model2 = conv_local(inputLayer_local)
    model2 = Dense(16,  kernel_size=5, strides=1, padding='same',
                   dilation_rate=1, activation='relu')(model2)
    model2 = Flatten()(model2)

    concatLayerQ = keras.layers.concatenate(
        [model1, model2], axis=1)

    denseLayerQ = Dense(512, activation='relu')(concatLayerQ)
    outputLayer = Dense(1, activation='sigmoid')(denseLayerQ)  # Output Layer

    model = Model(inputs=[inputLayer_local,
                          inputLayer_global], outputs=outputLayer)

    model.compile(loss='binary_crossentropy',
                  optimizer='adam', metrics=['accuracy'])

    return model
