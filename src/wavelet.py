from tensorflow.keras import layers
import os
import numpy as np
from kymatio.keras import Scattering1D
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf

if __name__ == "__main__":

    data_global = np.loadtxt('data/global_movavg.csv', delimiter=',')
    global_X = data_global[0:, 1:-1]  # Input
    global_Y = data_global[0:, -1]  # Labels

    global_X = data_global[0:, 1:-1]  # Input
    global_Y = data_global[0:, -1]  # Labels
    X_train_global, X_test_global, y_train_global, y_test_global = train_test_split(
        global_X, global_Y, test_size=0.2, random_state=1)

    scaler_global = MinMaxScaler(feature_range=(0, 1))  # Scale Values
    X_train_global = scaler_global.fit_transform(X_train_global)
    X_test_global = scaler_global.transform(X_test_global)

    J = 8
    Q = 12

    x_in = layers.Input(shape=(2001))
    x = Scattering1D(J, Q=Q)(x_in)
    x = layers.Lambda(lambda x: x[..., 1:, :])(x)
    x = layers.Lambda(lambda x: tf.math.log(tf.abs(x) + 1e-6))(x)
    x = layers.GlobalAveragePooling1D(data_format='channels_first')(x)
    x = layers.BatchNormalization(axis=1)(x)
    x_out = layers.Dense(1, activation='sigmoid')(x)
    model = tf.keras.models.Model(x_in, x_out)
    model.summary()
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    model.fit(X_train_global, y_train_global, epochs=50,
              batch_size=64, validation_split=0.2)
