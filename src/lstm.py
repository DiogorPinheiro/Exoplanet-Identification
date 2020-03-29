import numpy as np
from comet_ml import Experiment, Optimizer
import tensorflow as tf
from sklearn import preprocessing
from keras.models import Sequential
from keras.layers import Dense, Input, concatenate, Flatten, Dropout, PReLU, BatchNormalization, Activation, LSTM, CuDNNLSTM
from keras.initializers import Ones, Orthogonal, VarianceScaling, Zeros
from keras.callbacks import EarlyStopping
from keras.models import Model
from keras import optimizers
from sklearn.preprocessing import MinMaxScaler
import time as t
from sklearn.metrics import roc_auc_score, recall_score, precision_score
from sklearn.model_selection import StratifiedKFold
from keras import backend as K
from sklearn.metrics import roc_auc_score, recall_score, precision_score, f1_score


from dataFunctions import dataInfo

CSV_FILE = "/home/jcneves/Documents/Identifying-Exoplanets-Using-ML/src/q1_q17_dr24_tce_2020.01.28_08.52.13.csv"
DATA_DIRECTORY = "/home/jcneves/Documents/keplerData"

def auc_roc(y_true, y_pred):
    # any tensorflow metric
    value, update_op = tf.contrib.metrics.streaming_auc(y_pred, y_true)

    # find all variables created for this metric
    metric_vars = [i for i in tf.local_variables() if 'auc_roc' in i.name.split('/')[1]]

    # Add metric variables to GLOBAL_VARIABLES collection.
    # They will be initialized for new session.
    for v in metric_vars:
        tf.add_to_collection(tf.GraphKeys.GLOBAL_VARIABLES, v)

    # force to update metric values
    with tf.control_dependencies([update_op]):
        value = tf.identity(value)
        return value

def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

def training(model, train_global, train_local, yglobal,ylocal , nb_cv = 10, batch_size = 5, nb_epochs = 10):

    kfold = StratifiedKFold(n_splits=nb_cv, shuffle=True, random_state=7)
    #print(type(train_global))
    #d = {'a':pd.Series(train_global),'b':pd.Series(train_global)}
    #X = pd.DataFrame(data=d,index=[0])

    cvscores = []

    for train_index, valid_index in kfold.split(train_global, ylocal):
        X_train_fold = train_global[train_index]
        X_valid_fold = train_global[valid_index]
        y_train_fold = ylocal[train_index]
        y_valid_fold = ylocal[valid_index]

        # Reshape data to 3D input
        #X_train_fold = np.expand_dims(X_train_fold, axis=2)
        #X_valid_fold=np.expand_dims(X_valid_fold, axis=2)

        model.fit(X_train_fold, y_train_fold, batch_size=32, epochs=50,
                  validation_data=(X_valid_fold, y_valid_fold),
                  callbacks=[EarlyStopping(monitor='val_auc_roc', min_delta=0, patience=0, verbose=1, mode='max')])
        score = model.evaluate(X_valid_fold, y_valid_fold, verbose=0)[1]

        #score, acc = model.evaluate(X_valid_fold, y_valid_fold, batch_size, verbose=0)

        Y_score = model.predict(X_valid_fold)
        Y_predict = model.predict_classes(X_valid_fold)

        auc = roc_auc_score(y_valid_fold, Y_score)
        recall = recall_score(y_valid_fold, Y_predict)
        precision = precision_score(y_valid_fold, Y_predict)
        f1 = f1_score(y_valid_fold, Y_predict)

        print('\n')
        print('ROC/AUC Score: ', auc)
        print('Precision: ', precision)
        print('Recall: ', recall)
        print('F1: ', f1_score)

        cvscores.append(auc)
    return np.mean(cvscores)


def model_creator(train_X_global,ls_units, dense_units, dropout_d, dropout_l, learn_rate, momentum):
    input = Input(shape=(train_X_global.shape[1], 1))

    model = LSTM(units=ls_units,return_sequences=True)(input)
    #model = LSTM(units=ls_units)(input)
    #model = VarianceScaling(scale=1.0,mode='fan_avg', distribution='uniform', seed=None)(model)
    #model = Zeros()(model)
    #model=Orthogonal(gain=1.0, seed=None)(model)
    model = BatchNormalization()(model)
    #model=Ones()(model)
    #model = Zeros()(model)
    #model = Zeros()(model)
    #model = Ones()(model)
    model = Dropout(dropout_l)(model)
    model =PReLU()(model)
    #model = Zeros()(model)
    #model=Flatten()(model)

    model = Dense(units=dense_units)(model)
    #model = VarianceScaling(scale=1.0,mode='fan_avg', distribution='uniform')(model)
    #model = Zeros()(model)
    model = Dropout(dropout_d)(model)
    model =PReLU()(model)
    #model = Zeros()(model)
    model = Dense(units=dense_units)(model)
    #model = VarianceScaling(scale=1.0,mode='fan_avg', distribution='uniform')(model)
    #model = Zeros()(model)
    model = PReLU()(model)
    model = Flatten()(model)

    out = Dense(1, activation='sigmoid')(model)

    model = Model(inputs=input, outputs=out)

    opt = optimizers.SGD(lr=0.01*learn_rate, decay=0.0001, momentum=momentum, nesterov=True)
    model.compile(loss='binary_crossentropy', optimizer=opt,  metrics=['accuracy',f1_m,precision_m, recall_m])
    return model

def fits(train_X_global, train_Y_global, val_X_global, val_Y_global, test_X_global, test_Y_global, epochs, batch_size, ls_units, dense_units,dropout_d, dropout_l, learn_rate, momentum, train_X_local, train_Y_local, val_X_local, val_Y_local, test_X_local, test_Y_local ):
    model = model_creator(train_X_global,ls_units, dense_units, dropout_d, dropout_l, learn_rate, momentum )

    # Local or Global View
    model.fit(train_X_global, train_Y_global, batch_size=batch_size, epochs=epochs,
              validation_data=(val_X_global, val_Y_global),
              callbacks=[EarlyStopping(monitor='val_loss', min_delta=1, patience=10, verbose=1, mode='min')])
    score = model.evaluate(test_X_global, test_Y_global, verbose=1)[1]

    # Local and Global View
    #model.fit([train_X_global,train_X_local], train_Y_global, batch_size=batch_size, epochs=epoch,validation_data=([val_X_global,val_X_local], val_Y_global), callbacks=[EarlyStopping(monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='auto')])
    #score = model.evaluate([test_X_global,test_X_local], test_Y_global, verbose=0)[1]

    return score



def main():
    start = t.time()

    #experiment = Experiment("hMRp4uInUqRHs0pHtHFTl6jUL")

    # Data For The Sequential 1D-LSTM
    data_local = np.loadtxt('neural_input_local.csv', delimiter=',')
    local_X = data_local[0:, 1:-1]  # Input
    local_Y = data_local[0:, -1]  # Labels

    data_global = np.loadtxt('neural_input_global.csv', delimiter=',')
    global_X = data_global[0:, 1:-1]  # Input
    global_Y = data_global[0:, -1]  # Labels

    # Separate Data
    train_X_local, val_X_local, test_X_local = np.split(local_X, [int(.8 * len(local_X)), int(0.9 * len(local_X))])  # Training = 80%, Validation = 10%, Test = 10%
    train_Y_local, val_Y_local, test_Y_local = np.split(local_Y, [int(.8 * len(local_Y)), int(0.9 * len(local_Y))])
    # print("Total: {} ; Training: {} ; Evaluation: {} ; Test: {}".format(len(local_X),len(train_X_local),len(val_X_local),len(test_X_local)))
    scaler_local = MinMaxScaler(feature_range=(0, 1))  # Scale Values
    train_X_local = scaler_local.fit_transform(train_X_local)
    val_X_local = scaler_local.transform(val_X_local)
    test_X_local = scaler_local.transform(test_X_local)

    train_X_global, val_X_global, test_X_global = np.split(global_X, [int(.8 * len(global_X)), int(0.9 * len(global_X))])  # Training = 80%, Validation = 10%, Test = 10%
    train_Y_global, val_Y_global, test_Y_global = np.split(global_Y, [int(.8 * len(global_Y)), int(0.9 * len(global_Y))])
    # print("Total: {} ; Training: {} ; Evaluation: {} ; Test: {}".format(len(global_X),len(train_X_global),len(val_X_global),len(test_X_global)))
    scaler_global = MinMaxScaler(feature_range=(0, 1))  # Scale Values
    train_X_global = scaler_global.fit_transform(train_X_global)
    val_X_global = scaler_global.transform(val_X_global)
    test_X_global = scaler_global.transform(test_X_global)

    # Shape Data
    train_X_global = np.expand_dims(train_X_global, axis=2)
    val_X_global = np.expand_dims(val_X_global, axis=2)
    test_X_global = np.expand_dims(test_X_global, axis=2)
    train_X_local = np.expand_dims(train_X_local, axis=2)
    val_X_local = np.expand_dims(val_X_local, axis=2)
    test_X_local = np.expand_dims(test_X_local, axis=2)
    '''
    batch_size = 32
    epochs = 20
    ls_units = 5
    dense_units = 64
    dropout_d = 0.2
    dropout_l = 0.2
    learn_rate = 0.03
    momentum = 0.2
    params = {'batch_size': batch_size,
              'epochs': epochs,
              'ls_units': ls_units,
              'dense_units': dense_units,
              'dropout_d': dropout_d,
              'dropout_l': dropout_l,
              'learn_rate': learn_rate,
              'momentum': momentum,
              }

    config = {
        "algorithm": "bayes",
        "name": "Optimize LSTM Network",
        "spec": {"maxCombo": 0, "objective": "minimize", "metric": "loss"},
        "parameters": {
            "batch_size": {"type": "discrete", "values":[16,32]},
            "epochs": {"type": "integer", "min": 10, "max": 50},
            "ls_units": {"type": "discrete", "values":[2,5,10,15]},
            "dense_units": {"type": "discrete","values": [32,64,128]},
            "dropout_d": {"type": "float","scalingType":"uniform", "min": 0.0, "max": 0.5},
            "dropout_l": {"type": "float", "scalingType": "uniform", "min": 0.0, "max": 0.5},
            "learn_rate": {"type": "float","scalingType":"loguniform", "min": 0.01, "max": 0.5},
            "momentum": {"type": "float","scalingType":"uniform", "min":0.0, "max": 0.5},
        },
        "trials": 1,
    }



    opt = Optimizer(config, api_key="hMRp4uInUqRHs0pHtHFTl6jUL", project_name="lstm1")

    for experiment in opt.get_experiments():
        epochs = experiment.get_parameter("epochs")
        batch_size = experiment.get_parameter("batch_size")
        ls_units = experiment.get_parameter("ls_units")
        dense_units = experiment.get_parameter("dense_units")
        dropout_d = experiment.get_parameter("dropout_d")
        dropout_l = experiment.get_parameter("dropout_l")
        learn_rate = experiment.get_parameter("learn_rate")
        momentum = experiment.get_parameter("momentum")

        acc = fits(train_X_global, train_Y_global, val_X_global, val_Y_global, test_X_global, test_Y_global,
                  epochs, batch_size, ls_units, dense_units, dropout_d, dropout_l, learn_rate, momentum, train_X_local, train_Y_local, val_X_local, val_Y_local, test_X_local, test_Y_local )
        # Reverse the score for minimization
        experiment.log_metric("loss", acc)

    experiment.log_parameters(params)
    '''
    model = model_creator(train_X_global,10, 64, 0.298, 0.298, 0.00643199565237, 0.25 )
    score = training(model, train_X_global, train_X_local, train_Y_global, train_Y_local, 10, 32, 50)

    # Train And Evaluate Model
    #model = model_creator(train_X_global)
    #model.fit(train_X_global, train_Y_global, batch_size=16, epochs=43, validation_data=(val_X_global, val_Y_global), callbacks=[EarlyStopping(monitor='roc_auc', min_delta=0, patience=2, verbose=1, mode='max')])
    #score = model.evaluate(test_X_global, test_Y_global, verbose=0)[1]
    #print("Test Accuracy = {}".format(score))

main()