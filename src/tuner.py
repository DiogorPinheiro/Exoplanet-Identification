import numpy as np
import tensorflow as tf
from tensorflow import keras
from kerastuner import Hyperband
from kerastuner.engine.hyperparameters import HyperParameters
#import shuffle
from keras import optimizers
from sklearn.model_selection import train_test_split
from keras.layers import Dense, Input, concatenate, Flatten, Dropout, PReLU, BatchNormalization, Activation, GaussianNoise, MaxPooling1D
from keras.layers.convolutional import Conv1D
from keras.callbacks import EarlyStopping, History, CSVLogger


from training import f1_m, precision_m, recall_m, mainEvaluate, auc_roc

data_global = np.loadtxt(
    'global_train.csv', delimiter=',')
global_X = data_global[0:, 0:-1]  # Input
global_Y = data_global[0:, -1]  # Labels

# Suffle Data (Only For Shallue Datasets)
#global_X, global_Y = shuffle(global_X, global_Y)
X_train_global, X_test_global, y_train_global, y_test_global = train_test_split(
    global_X, global_Y, test_size=0.2, random_state=1)

# Shape Data
X_train_global_shaped = np.expand_dims(X_train_global, axis=2)
X_test_global_shaped = np.expand_dims(X_test_global, axis=2)


def build_model(hp):
    inputs = tf.keras.Input(shape=(X_test_global_shaped.shape[1], 1))
    x = inputs
    for i in range(hp.Int('conv_blocks', 1, 5, default=3)):
        filters = hp.Int('filters_' + str(i), 8, 256, step=32)
        for _ in range(2):
            x = tf.keras.layers.Convolution1D(
                filters, hp.Int('kernel_size', 1, 9, default=3), padding='same')(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.Dropout(
                hp.Float('dropout', 0, 0.4, step=0.05, default=0.2))(x)
            x = tf.keras.layers.ReLU()(x)
        if hp.Choice('pooling_' + str(i), ['avg', 'max']) == 'max':
            x = tf.keras.layers.MaxPool1D()(x)
        else:
            x = tf.keras.layers.AvgPool1D()(x)
    for f in range(hp.Int('dense_blocks', 0, 5, default=3)):
        x = tf.keras.layers.GlobalAvgPool1D()(x)
        x = tf.keras.layers.Dense(
            hp.Int('hidden_size', 32, 254, step=10, default=50),
            activation='relu')(x)
        x = tf.keras.layers.Dropout(
            hp.Float('dropout', 0, 0.5, step=0.1, default=0.5))(x)
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)

    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer=tf.keras.optimizers.SGD(hp.Float('lr', 0, 0.1, step=0.001, default=0.01), hp.Float('momentum', 0, 0.5, step=0.01, default=0.1)),
                  loss='binary_crossentropy',
                  metrics=['accuracy', f1_m, precision_m, recall_m, tf.keras.metrics.AUC()])
    return model


if __name__ == "__main__":

    tuner = Hyperband(build_model, objective='val_loss',
                      max_epochs=40, project_name='num4', seed=4)
    tuner.search(X_train_global_shaped, y_train_global,
                 epochs=40, validation_data=(X_test_global_shaped, y_test_global), callbacks=[EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1, mode='auto')])
    best_model = tuner.get_best_models(1)[0]
    best_model.summary()
    best_param = tuner.get_best_hyperparameters(1)[0]
    print(best_param.values)

    best_model.save('model2.hdf5')
