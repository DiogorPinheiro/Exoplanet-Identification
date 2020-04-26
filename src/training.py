import numpy as np
import tensorflow as tf
import keras
from time import time
from sklearn.metrics import log_loss
from keras.callbacks import EarlyStopping, History, CSVLogger
from keras import backend as K
from sklearn.metrics import roc_auc_score, recall_score, precision_score, f1_score, classification_report
from sklearn.model_selection import StratifiedKFold
from keras import objectives
from keras.callbacks import TensorBoard

from utilities import concatenate, writeToFile

# ------------------------------ Evaluation Metrics ----------------------------------------


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


def auc_roc(y_true, y_pred):
    # any tensorflow metric
    value, update_op = tf.contrib.metrics.streaming_auc(y_pred, y_true)

    # find all variables created for this metric
    metric_vars = [i for i in tf.local_variables(
    ) if 'auc_roc' in i.name.split('/')[1]]

    # Add metric variables to GLOBAL_VARIABLES collection.
    # They will be initialized for new session.
    for v in metric_vars:
        tf.add_to_collection(tf.GraphKeys.GLOBAL_VARIABLES, v)

    # force to update metric values
    with tf.control_dependencies([update_op]):
        value = tf.identity(value)
        return value


class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))


def customLoss(ytrue, ypred):
    x = keras.losses.binary_crossentropy(ytrue, ypred, from_logits=False)
    # x = K.print_tensor(x, message="losses = ")
    return x


def fn_loss(ytrue, ypred):
    return ypred


class customMetrics(keras.callbacks.Callback):
    def on_batch_end(self, batch, logs=None):
        # self.losses.append(logs.get('customLoss'))
        for k in logs:
            if k.endswith('fn_loss'):
                print("Valor {}".format(logs[k]))


def logloss(true_label, predicted, eps=1e-15):
    p = np.clip(predicted, eps, 1 - eps)
    if true_label == 1:
        return -np.log(p)
    else:
        return -np.log(1 - p)

# -------------------------------- Evaluation Process --------------------------------------------


def evaluateDual(model, x_agg, y, splits, batch, epochs, type, filename, X_test_global, X_test_local, y_test):
    kfold = StratifiedKFold(n_splits=splits, shuffle=True,
                            random_state=7)  # Cross-Validation

    tensorboard = TensorBoard(
        log_dir="logs/{}".format(time()))  # Visualization

    # history = History()
    history = LossHistory()
    cvscores = []

    for train_index, valid_index in kfold.split(x_agg, y):
        # print("Train index {}".format(train_index))
        # print("Valid index {}".format(valid_index))

        X_train_fold = x_agg[train_index]
        X_valid_fold = x_agg[valid_index]
        y_train_fold = y[train_index]
        y_valid_fold = y[valid_index]

        x_train_global = X_train_fold[0:, :2001]
        x_train_local = X_train_fold[0:, 2001:]
        x_valid_global = X_valid_fold[0:, :2001]
        x_valid_local = X_valid_fold[0:, 2001:]

        # print(("x_train_gl {} ; x_val_gl {} ; x_train_l {} ; x_val_l {}").format(x_train_global.shape, x_valid_global.shape, x_train_local.shape, x_valid_local.shape))

        model.fit([x_train_global, x_train_local], y_train_fold, batch_size=batch, epochs=epochs, validation_data=([x_valid_global, x_valid_local], y_valid_fold),
                  callbacks=[EarlyStopping(monitor='val_auc_roc', min_delta=0, patience=10, verbose=1, mode='max'), history, tensorboard])
        score = model.evaluate(
            [x_valid_global, x_valid_local], y_valid_fold, verbose=0)[1]

        # Print Results
        print(len(train_index))
        print(history.losses)
        print(len(history.losses))
        print(len(x_agg))

        Y_score = model.predict([x_valid_global, x_valid_local])
        Y_classes = Y_score.argmax(axis=-1)

        '''
        if (type == 'sequential'):
            Y_predict = model.predict_classes([x_valid_global, x_valid_local],)
            recall = recall_score(y_valid_fold, Y_predict)
            precision = precision_score(y_valid_fold, Y_predict)
            f1 = f1_score(y_valid_fold, Y_predict)

            print('Precision: ', precision)
            print('Recall: ', recall)
            print('F1: ', f1_score)
        else:
            recall = recall_score(y_valid_fold, Y_classes)
            precision = precision_score(y_valid_fold, Y_classes)
            f1 = f1_score(y_valid_fold, Y_classes)
        '''
        # print(classification_report(y_valid_fold, Y_classes))
        auc = roc_auc_score(y_valid_fold, Y_score)

        print('\n')
        print('ROC/AUC Score: ', auc)

        cvscores.append(auc)

    # Get Loss Log For Each Sample
    Y_score_log = model.predict([X_test_global, X_test_local])
    print("Test Length {}".format(len(y_test)))
    print("Prediction Score Length {}".format(len(Y_score_log)))
    tens = K.eval(customLoss(K.variable(y_test), K.variable(Y_score_log)))

    # Save Model
    model.save(filename)
    print("Saved model to disk")

    return np.mean(cvscores), history.losses, tens


def evaluateDualfnn(model, x_agg, y, splits, batch, epochs, type, filename, X_test_global, X_test_local, y_test):
    kfold = StratifiedKFold(n_splits=splits, shuffle=True, random_state=7)

    tensorboard = TensorBoard(
        log_dir="logs/{}".format(time()))  # Visualization

    # history = History()
    history = LossHistory()
    cvscores = []

    for train_index, valid_index in kfold.split(x_agg, y):
        # print("Train index {}".format(train_index))
        # print("Valid index {}".format(valid_index))

        X_train_fold = x_agg[train_index]
        X_valid_fold = x_agg[valid_index]
        y_train_fold = y[train_index]
        y_valid_fold = y[valid_index]

        x_train_global = X_train_fold[0:, :2001]
        x_train_local = X_train_fold[0:, 2001:]
        x_valid_global = X_valid_fold[0:, :2001]
        x_valid_local = X_valid_fold[0:, 2001:]

        # print(("x_train_gl {} ; x_val_gl {} ; x_train_l {} ; x_val_l {}").format(x_train_global.shape, x_valid_global.shape, x_train_local.shape, x_valid_local.shape))

        model.fit([x_train_local, x_train_global], y_train_fold, batch_size=batch, epochs=epochs, validation_data=([x_valid_local, x_valid_global], y_valid_fold),
                  callbacks=[EarlyStopping(monitor='val_auc_roc', min_delta=0, patience=10, verbose=1, mode='max'), history, tensorboard])
        score = model.evaluate(
            [x_valid_local, x_valid_global], y_valid_fold, verbose=0)[1]

        print(len(train_index))
        print(history.losses)
        print(len(history.losses))
        print(len(x_agg))

        Y_score = model.predict([x_valid_local, x_valid_global])
        Y_classes = Y_score.argmax(axis=-1)

        if (type == 'sequential'):
            Y_predict = model.predict_classes([x_valid_global, x_valid_local],)
            recall = recall_score(y_valid_fold, Y_predict)
            precision = precision_score(y_valid_fold, Y_predict)
            f1 = f1_score(y_valid_fold, Y_predict)

            print('Precision: ', precision)
            print('Recall: ', recall)
            print('F1: ', f1_score)

        # (classification_report(y_valid_fold, Y_classes))
        auc = roc_auc_score(y_valid_fold, Y_score)

        print('\n')
        print('ROC/AUC Score: ', auc)

        cvscores.append(auc)

    # Get Loss Log For Each Sample
    Y_score_log = model.predict([X_test_local, X_test_global])
    print("Test Length {}".format(len(y_test)))
    print("Prediction Score Length {}".format(len(Y_score_log)))
    tens = K.eval(customLoss(K.variable(y_test), K.variable(Y_score_log)))

    # Save Model
    model.save(filename)
    print("Saved model to disk")

    return np.mean(cvscores), history.losses, tens


def evaluateSingle(model, X, y, splits, batch, epoch, type, filename, X_test, y_test):
    kfold = StratifiedKFold(n_splits=splits, shuffle=True, random_state=7)

    tensorboard = TensorBoard(
        log_dir="logs/{}".format(time()))  # Visualization

    # history = History()
    history = LossHistory()
    cvscores = []
    csv_logger = CSVLogger('log.csv', append=True, separator=',')
    for train_index, valid_index in kfold.split(X, y):
        # print("Train index {}".format(train_index))
        # print("Valid index {}".format(valid_index))

        X_train_fold = X[train_index]
        X_valid_fold = X[valid_index]
        y_train_fold = y[train_index]
        y_valid_fold = y[valid_index]

        # print(("x_train_gl {} ; x_val_gl {} ; x_train_l {} ; x_val_l {}").format(x_train_global.shape,x_valid_global.shape, x_train_local.shape,x_valid_local.shape))
        model.fit(X_train_fold, y_train_fold, batch_size=batch, epochs=epoch, validation_data=(X_valid_fold, y_valid_fold),
                  callbacks=[EarlyStopping(monitor='val_auc_roc', min_delta=0, patience=10, verbose=1, mode='max'), history, csv_logger, customMetrics(), tensorboard])
        score = model.evaluate(
            X_valid_fold, y_valid_fold, verbose=0)[1]

        print(len(train_index))
        print(history.losses)
        print(len(history.losses))

        Y_score = model.predict(X_valid_fold)
        Y_classes = Y_score.argmax(axis=-1)

        print("Validation Length {}".format(len(valid_index)))
        print("Prediction Score Length {}".format(len(Y_score)))
        tens = K.eval(customLoss(K.variable(
            y_valid_fold), K.variable(Y_score)))

        if (type == 'sequential'):
            Y_predict = model.predict_classes(X_valid_fold)
            recall = recall_score(y_valid_fold, Y_predict)
            precision = precision_score(y_valid_fold, Y_predict)
            f1 = f1_score(y_valid_fold, Y_predict)

            print('Precision: ', precision)
            print('Recall: ', recall)
            print('F1: ', f1_score)

        # print(classification_report(y_valid_fold, Y_classes))
        auc = roc_auc_score(y_valid_fold, Y_score)

        print('\n')
        print('ROC/AUC Score: ', auc)

        cvscores.append(auc)

    # Get Loss Log For Each Sample
    Y_score_log = model.predict(X_test)
    print("Test Length {}".format(len(y_test)))
    print("Prediction Score Length {}".format(len(Y_score_log)))
    tens = K.eval(customLoss(K.variable(y_test), K.variable(Y_score_log)))

    score = model.evaluate(X_test, y_test, verbose=0)
    print("Test -> %s: %.2f%%" % (model.metrics_names[1], score[1]*100))

    # Save Model
    # model.save(filename)
    # print("Saved model to disk")

    return np.mean(cvscores), history.losses, tens


def evaluateSimple(model, X_train, y_train, X_test, y_test, splits):
        # model.fit(X_train, y_train)
        # score = model.score(X_test, y_test)
    kfold = StratifiedKFold(n_splits=splits, shuffle=False, random_state=7)

    # history = History()
    history = LossHistory()
    cvscores = []
    for train_index, valid_index in kfold.split(X_train, y_train):
        # print("Train index {}".format(train_index))
        # print("Valid index {}".format(valid_index))

        # writeToFile("test2.txt",train_index)

        X_train_fold = X_train[train_index]
        X_valid_fold = X_train[valid_index]
        y_train_fold = y_train[train_index]
        y_valid_fold = y_train[valid_index]

        model.fit(X_train, y_train)

        score = model.score(X_valid_fold, y_valid_fold)

        Y_score = model.predict(X_valid_fold)
        Y_classes = Y_score.argmax(axis=-1)

        # print("Validation Length {}".format(len(valid_index)))
        # print("Prediction Score Length {}".format(len(Y_score)))
        # tens = K.eval(customLoss(K.variable(
        #    y_valid_fold), K.variable(Y_score)))

        # print(classification_report(y_valid_fold, Y_classes))
        auc = roc_auc_score(y_valid_fold, Y_score)

        print('\n')
        print('ROC/AUC Score: ', auc)
        cvscores.append(auc)

    # Get Loss Log For Each Sample
    Y_score_log = model.predict(X_test)
    Y_classes = Y_score.argmax(axis=-1)
    print(Y_score_log)

    # tens = []
    # for i in range(len(Y_score_log)):
    #    tens.append(log_loss(y_test[i], Y_score_log[i]))
    print(log_loss(y_test, Y_score_log))
    tens = [logloss(x, y) for (x, y) in zip(y_test, Y_score_log)]
    print(len(tens))
    print(len(y_test))
    print("Loss {}".format(np.mean(tens)))
    # print("Test Length {}".format(len(y_test)))
    # print("Prediction Score Length {}".format(len(Y_score)))
    # tens = K.eval(customLoss(K.variable(y_test), K.variable(Y_score)))
    return np.mean(cvscores), tens


def mainEvaluate(option, model, train_global, train_local, test_global, test_local, y_train, y_test, nb_cv, epoch, batch, splits, type, filename):
    if(option == 'dual'):
        agg = concatenate(train_global, train_local)
        # agg = np.expand_dims(agg, axis=2)
        print(agg.shape)
        return evaluateDual(model, agg, y_train,
                            splits, batch, epoch, type, filename, test_global, test_local, y_test)
    elif (option == 'dual-fnn'):
        agg = concatenate(train_global, train_local)
        # agg = np.expand_dims(agg, axis=2)
        print(agg.shape)
        return evaluateDualfnn(model, agg, y_train,
                               splits, batch, epoch, type, filename, test_global, test_local, y_test)
    elif(option == 'single-global'):
        return evaluateSingle(model, train_global, y_train,
                              splits, batch, epoch, type, filename, test_global, y_test)
    elif(option == 'single-local'):
        return evaluateSingle(model, train_local, y_train,
                              splits, batch, epoch, type, filename, test_local, y_test)
    elif(option == 'simple-global'):
        return evaluateSimple(model, train_global, y_train, test_global, y_test, splits)
    elif(option == 'simple-local'):
        return evaluateSimple(model, train_local, y_train, test_local, y_test, splits)