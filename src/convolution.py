import numpy as np
from comet_ml import Experiment, Optimizer
from sklearn import preprocessing
from keras.models import Sequential
from keras.layers import Dense, Input, concatenate, Flatten, Dropout, PReLU, BatchNormalization, Activation
from keras.layers.convolutional import Conv1D
from keras.layers import MaxPooling1D
from keras.callbacks import EarlyStopping
from keras.models import Model
from keras import optimizers
from sklearn.preprocessing import MinMaxScaler
import time as t
from sklearn.metrics import roc_auc_score, recall_score, precision_score


from dataFunctions import dataInfo

CSV_FILE = "/home/jcneves/Documents/Identifying-Exoplanets-Using-ML/src/q1_q17_dr24_tce_2020.01.28_08.52.13.csv"
DATA_DIRECTORY = "/home/jcneves/Documents/keplerData"

def getKepids(table):
    '''
        Get All Kepids In The CSV File

        Output: List of Kepids Numbers (List of Int)
    '''
    return dataInfo.listKepids(table)

def getCSVData():
    '''
        Call dataCSV Function To Get The Data Of CSV File

        Output: Pandas Dataframe
    '''
    return dataInfo.dataCSV(CSV_FILE)

def normalizeData(data):
    '''
        Read CSV File And Normalize Each Column (Except Kepid and Label)

        Replaces Values In CSV File With Their Normalized Version
    '''

    normalized_results=[]

    i=1
    for col in data:
        x_array = np.array(col)
        #print("Array:{}".format(x_array))
        result = preprocessing.normalize([x_array])
        #print(val)
        normalized_results.append(result)


    return normalized_results

def bothViewsCNN(x_train_local, x_train_global, train_X_local, train_X_global, train_Y_local, test_X_local, test_X_global, test_Y_local ):
    # CNN Model
    inputLayer_local = Input(shape=x_train_local.shape)
    inputLayer_global = Input(shape=x_train_local.shape)

    conv_local = Conv1D(201, 10, strides=1, input_shape=x_train_local.shape, padding='same', dilation_rate=1,
                        activation='relu')
    conv_global = Conv1D(2001, 10, strides=1, input_shape=x_train_local.shape, padding='same', dilation_rate=1,
                         activation='relu')

    convQ1 = conv_local(inputLayer_local)  # Disjoint Conv Layer
    poolLayerQ1 = MaxPooling1D(pool_size=5, strides=1, padding='valid')(convQ1)
    convQ2 = conv_global(inputLayer_global)
    poolLayerQ2 = MaxPooling1D(pool_size=5, strides=1, padding='valid')(convQ2)

    concatLayerQ = concatenate([inputLayer_local, inputLayer_global], axis=1)  # Concatenate Layer
    flatLayerQ = Flatten()(concatLayerQ)
    denseLayerQ = Dense(200, activation='relu')(flatLayerQ)

    outputLayer = Dense(1, activation='sigmoid')(denseLayerQ)  # Output Layer

    model = Model(inputs=[inputLayer_local, inputLayer_global], outputs=outputLayer)
    opt = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    # Training The Model
    model.fit([train_X_local, train_X_global], train_Y_local, epochs=20, batch_size=128)
    score = model.evaluate([test_X_local, test_X_global], test_Y_local, batch_size=128)

def training(model, X_train, y_train, X_val, y_val, nb_cv = 10, batch_size = 5, nb_epochs = 10):
    # define 10-fold cross validation test harness
    kfold = StratifiedKFold(n_splits=nb_cv, shuffle=True, random_state=7)

    cvscores = []

    for train_index, valid_index in kfold.split(X_train, y_train):
        X_train_fold = X_train[train_index]
        X_valid_fold = X_train[valid_index]
        y_train_fold = y_train[train_index]
        y_valid_fold = y_train[valid_index]

        # Reshape data to 3D input
        #X_train_fold = np.expand_dims(X_train_fold, axis=2)
        #X_valid_fold=np.expand_dims(X_valid_fold, axis=2)

        model.fit(X_train_fold, y_train_fold, batch_size=batch_size, epochs=nb_epochs, verbose=2)

        score, acc = model.evaluate(X_valid_fold, y_valid_fold, batch_size, verbose=0)

        Y_score = model.predict(X_valid_fold)

        auc = roc_auc_score(y_valid_fold, Y_score)
        #recall = recall_score(y_valid_fold, Y_predict)
        #precision = precision_score(y_valid_fold, Y_predict)

        print('\n')
        print('Acc: ', acc)
        print('ROC/AUC Score: ', auc)
        #print('Precision: ', precision)
        #print('Recall: ', recall)
        print('\n')

        cvscores.append(auc)

def functionalCNN(x_train_global):
    var = Input(shape=(x_train_global.shape[1], 1))
    model = Conv1D(filters=128, kernel_size=3, padding='same')(var)
    model = BatchNormalization()(model)
    model = MaxPooling1D(pool_size=2, strides=2)(model)
    model = Dropout(0.20)(model)
    model = Activation('relu')(model)
    model = Conv1D(filters=128, kernel_size=3, padding='same')(model)
    model = BatchNormalization()(model)
    model = MaxPooling1D(pool_size=2, strides=2)(model)
    model = Dropout(0.20)(model)
    model = Activation('relu')(model)
    model = Flatten()(model)
    model = Dense(64)(model)
    model = Dropout(0.20)(model)
    model = Activation('relu')(model)
    out = Dense(1, activation='sigmoid')(model)
    model = Model(inputs=var, outputs=out)
    return model

def fit(experiment, train_X_global, train_Y_global, val_X_global, val_Y_global, test_X_global, test_Y_global, epoch, batch_size,lay1_filters,l1_kernel_size,pool_size,strides,conv_dropout,lay2_filters,l2_kernel_size,dense_f,dense_dropout,x_train_global):
    model = seqModelCNN(lay1_filters,l1_kernel_size,pool_size,strides,conv_dropout,lay2_filters,l2_kernel_size,dense_f,dense_dropout,x_train_global)
    '''    with experiment.train():
        history = model.fit(train_X_global, train_Y_global,
                            batch_size=batch_size,
                            epochs=epoch,
                            verbose=1,
                            validation_data=(val_X_global, val_Y_global),
                            callbacks=[
                                EarlyStopping(monitor='val_loss', min_delta=1e-4, patience=3, verbose=1, mode='auto')])

    # will log metrics with the prefix 'test_'
    with experiment.test():
        loss, accuracy = model.evaluate(test_X_global, test_Y_global)
        metrics = {
            'loss': loss,
            'accuracy': accuracy
        }
        experiment.log_metrics(metrics)

    experiment.log_dataset_hash(train_X_global)
    '''
    model.fit(train_X_global, train_Y_global, batch_size=batch_size, epochs=epoch, validation_data=(val_X_global, val_Y_global), callbacks=[EarlyStopping(monitor='val_loss',
                              min_delta=0,
                              patience=0,
                              verbose=0, mode='auto')]   )
    score = model.evaluate(test_X_global, test_Y_global, verbose=0)[1]

    return score

def seqModelCNN(lay1_filters,l1_kernel_size,pool_size,strides,conv_dropout,lay2_filters,l2_kernel_size,dense_f,dense_dropout,x_train_global):
    model = Sequential()
    model.add(Conv1D(filters=lay1_filters, kernel_size=l1_kernel_size, input_shape=(x_train_global.shape[1], 1),
                     padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=pool_size, strides=strides))
    model.add(Dropout(conv_dropout))
    model.add(Activation('relu'))
    model.add(Conv1D(filters=lay2_filters, kernel_size=l2_kernel_size, padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=pool_size, strides=strides))
    model.add(Dropout(conv_dropout))
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(dense_f))
    model.add(Dropout(dense_dropout))
    model.add(Activation('relu'))
    model.add(Dense(1, activation='sigmoid'))

    opt = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    
    return model

def main():
    start = t.time()

    experiment = experiment = Experiment("hMRp4uInUqRHs0pHtHFTl6jUL")


    table = getCSVData().drop_duplicates()
    kepids = getKepids(table).drop_duplicates().reset_index(drop=True)  # List of Kepids
    #dataReader.createFluxDatabase(table,kepids,DATA_DIRECTORY)

    # Data For The Sequential 1D-CNN
    data_local = np.loadtxt('neural_input_local.csv', delimiter=',')
    local_X = data_local[0:, 1:-1]  # Input
    local_Y = data_local[0:, -1]  # Labels
    scaler_local = MinMaxScaler(feature_range=(0, 1))   # Scale Values
    rescaled_local_X = scaler_local.fit_transform(local_X)

    data_global = np.loadtxt('neural_input_global.csv', delimiter=',')
    global_X = data_global[0:, 1:-1]  # Input
    global_Y = data_global[0:, -1]  # Labels
    scaler_global = MinMaxScaler(feature_range=(0, 1))  # Scale Values
    rescaled_global_X = scaler_global.fit_transform(global_X)

    # Separate Data
    train_X_local, val_X_local, test_X_local = np.split(rescaled_local_X, [int(.8 * len(rescaled_local_X)), int(0.9 * len(rescaled_local_X))])  # Training = 80%, Validation = 10%, Test = 10%
    train_Y_local, val_Y_local, test_Y_local = np.split(local_Y, [int(.8 * len(local_Y)), int(0.9 * len(local_Y))])
    #print("Total: {} ; Training: {} ; Evaluation: {} ; Test: {}".format(len(local_X),len(train_X_local),len(val_X_local),len(test_X_local)))

    train_X_global, val_X_global, test_X_global = np.split(rescaled_global_X, [int(.8 * len(rescaled_global_X)), int(0.9 * len(rescaled_global_X))])  # Training = 80%, Validation = 10%, Test = 10%
    train_Y_global, val_Y_global, test_Y_global = np.split(global_Y, [int(.8 * len(global_Y)), int(0.9 * len(global_Y))])
    #print("Total: {} ; Training: {} ; Evaluation: {} ; Test: {}".format(len(global_X),len(train_X_global),len(val_X_global),len(test_X_global)))

    # Shape Data
    #print(train_X_local.shape)
    x_train_local = np.expand_dims(train_X_local, axis=2)
    #print(x_train_local.shape)

    #print(train_X_global.shape)
    #print(x_train_global.shape)

    x_train_global =  np.expand_dims(train_X_global, axis=2)
    #bothViewsCNN(x_train_local, x_train_global, train_X_local, train_X_global, train_Y_local, test_X_local, test_X_global, test_Y_local)

    batch_size = 128
    epochs = 20
    lay1_filters=128
    lay2_filters = 128
    l1_kernel_size = 3
    l2_kernel_size = 3
    pool_size = 2
    conv_dropout=0.2
    dense_dropout=0.2
    strides = 2
    dense_f=64
    optimizer = 'adam'
    activation = 'relu'
    params = {'batch_size': batch_size,
              'epochs': epochs,
              'lay1_filters':lay1_filters,
              'lay2_filters': lay2_filters,
              'layer1_type': 'Dense',
              'conv_dropout':conv_dropout,
              'dense_dropout': dense_dropout,
              'l1_kernel_size':l1_kernel_size,
              'l2_kernel_size': l2_kernel_size,
              'dense_f':dense_f,
              'strides':strides,
              'pool_size':pool_size,
              'layer1_activation': activation,
              'optimizer': optimizer
              }


    #model = seqModelCNN(lay1_filters,l1_kernel_size,pool_size,strides,conv_dropout,lay2_filters,l2_kernel_size,dense_f,dense_dropout,x_train_global)

    #opt = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    #model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

    train_X_global = np.expand_dims(train_X_global, axis=2)
    val_X_global = np.expand_dims(val_X_global, axis=2)
    test_X_global = np.expand_dims(test_X_global, axis=2)
    #model.fit(train_X_global, train_Y_global)

    config = {
        "algorithm": "bayes",
        "name": "Optimize CNN Network",
        "spec": {"maxCombo": 30, "objective": "minimize", "metric": "loss"},
        "parameters": {
            "batch_size": {"type": "integer", "min": 20, "max": 254},
            "epochs":{"type": "integer", "min": 10, "max": 40},
            "lay1_filters": {"type": "integer", "min": 32, "max": 254},
            "lay2_filters": {"type": "integer", "min": 32, "max": 254},
            "l1_kernel_size": {"type": "integer", "min": 1, "max": 5},
            "l2_kernel_size": {"type": "integer", "min": 1, "max": 5},
            "pool_size": {"type": "integer", "min": 2, "max": 4},
            "conv_dropout": {"type": "float",  "min": 0.01, "max": 0.3},
            "dense_dropout": {"type": "float",  "min": 0.01, "max": 0.3},
            "strides": {"type": "integer", "min": 2, "max": 4},
            "dense_f": {"type": "integer", "min": 8, "max": 254},
        },
        "trials": 1,
    }

    opt = Optimizer(config, api_key="hMRp4uInUqRHs0pHtHFTl6jUL", project_name="CNN-2")

    for experiment in opt.get_experiments():
        epochs = experiment.get_parameter("epochs")
        batch_size = experiment.get_parameter("batch_size")
        lay1_filters = experiment.get_parameter("lay1_filters")
        l1_kernel_size = experiment.get_parameter("l1_kernel_size")
        l2_kernel_size = experiment.get_parameter("l2_kernel_size")
        pool_size = experiment.get_parameter("pool_size")
        strides = experiment.get_parameter("strides")
        conv_dropout = experiment.get_parameter("conv_dropout")
        lay2_filters = experiment.get_parameter("lay2_filters")
        dense_f = experiment.get_parameter("dense_f")
        dense_dropout = experiment.get_parameter("dense_dropout")

        acc = fit(experiment, train_X_global, train_Y_global, val_X_global, val_Y_global, test_X_global, test_Y_global, epochs, batch_size,lay1_filters,l1_kernel_size,pool_size,strides,conv_dropout,lay2_filters,l2_kernel_size,dense_f,dense_dropout,x_train_global)
        # Reverse the score for minimization
        experiment.log_metric("accuracy", acc)


    experiment.log_parameters(params)

    #training(model, train_X_global, train_Y_global, val_X_global, val_Y_global, nb_cv = 5, batch_size = 10, nb_epochs = 2)

    end = t.time()
    #print(end - start)

main()