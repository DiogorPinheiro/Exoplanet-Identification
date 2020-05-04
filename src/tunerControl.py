import numpy as np
import tensorflow as tf
from tensorflow import keras
from kerastuner import Hyperband
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
from kerastuner.engine.hyperparameters import HyperParameters


from hyperModels import CNNHyperModel, CNNTrial, FNNHyperModel, CNNTrial2

if __name__ == "__main__":

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

    input_shape = (X_test_global_shaped.shape[1], 1)
    #build_model = CNNHyperModel(input_shape, 1)
    #build_model = CNNTrial(input_shape, 1)
    build_model = CNNTrial2(1)
    tuner = Hyperband(build_model, objective='val_loss',
                      max_epochs=40, project_name='num4', seed=4)
    tuner.search(X_train_global_shaped, y_train_global,
                 epochs=40, validation_data=(X_test_global_shaped, y_test_global), callbacks=[EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1, mode='auto')])
    best_model = tuner.get_best_models(1)[0]
    best_model.summary()
    best_param = tuner.get_best_hyperparameters(1)[0]
    print(best_param.values)

    best_model.save('model2.hdf5')
