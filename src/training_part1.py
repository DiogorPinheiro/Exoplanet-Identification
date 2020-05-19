from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, auc, precision_score, recall_score, f1_score, roc_curve
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerLine2D
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer


def knnParametersTuning(train_X, train_Y, val_X, val_Y):
    '''
        Tune K-Nearest Neighbors Hyper-Parameters
        Trains And Tests Data Using The Area Under Curve As Evaluation Metric

        Input: X and Y Values Of The Training And Validation Sets
        Output: Figure With Comparison Of Parameters (Saved To The Directory)
    '''
    k = list(range(1, 30))
    train_results = []
    test_results = []
    for i in k:
        model = KNeighborsClassifier(n_neighbors=i)

        model.fit(train_X, train_Y)
        train_pred = model.predict(train_X)

        false_positive_rate, true_positive_rate, thresholds = roc_curve(
            train_Y, train_pred)    # Using Area Under Curve As Evaluation Metric
        roc_auc = auc(false_positive_rate, true_positive_rate)

        train_results.append(roc_auc)

        y_pred = model.predict(val_X)
        false_positive_rate, true_positive_rate, thresholds = roc_curve(
            val_Y, y_pred)
        roc_auc = auc(false_positive_rate, true_positive_rate)

        test_results.append(roc_auc)
    print(test_results)

    # Create Figure With Parameters Comparison
    line1, = plt.plot(k, train_results, 'b', label="Train AUC")
    line2, = plt.plot(k, test_results, 'r', label="Validation AUC")
    plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
    plt.ylabel('AUC Score')
    plt.xlabel('Power Parameter')
    plt.savefig('knnScores.png', bbox_inches='tight')


def logRegParameterTuning(train_X, train_Y, val_X, val_Y):
    '''
        Hyper-Parameter Tuning For The Logistic Regression Algorithm

        Input: X and Y Values Of The Training And Validation Sets
        Output: Prints Best Parameters Combination and Predicted Accuracy On The Training Data
    '''
    grid = {"C": np.logspace(-3, 3, 7),
            "penalty": ["l1", "l2"]}  # l1 lasso l2 ridge
    logreg = LogisticRegression()
    logreg_cv = GridSearchCV(logreg, grid, cv=10)
    logreg_cv.fit(train_X, train_Y)
    print("Tuned Hyperparameters :(best parameters) ", logreg_cv.best_params_)
    print("Accuracy :", logreg_cv.best_score_)
    # smaller C specify stronger regularization.


def svmParameterTuning(train_X, train_Y, val_X, val_Y):
    '''
            Hyper-Parameter Tuning For The Support Vector Machine Algorithm

            Input: X and Y Values Of The Training And Validation Sets
            Output: Prints Best Parameters Combination and Predicted Accuracy On The Training Data
        '''
    parameters = {'kernel': ('linear', 'rbf'), 'C': [1, 10]}
    svr = svm.SVC(probability=False)
    model = GridSearchCV(svr, parameters, n_jobs=4, verbose=5)
    model.fit(train_X, train_Y)
    print("Tuned Hyperparameters :(best parameters) ", model.best_params_)
    print("Accuracy :", model.best_score_)


def models(algorithm, train_X, train_Y, val_X, val_Y, test_X, test_Y):
    '''
        Choose Best HyperParameters, Trains And Evaluates The Algorithm

        Input: algorithm - Array Of String With The Names Of All Used Algorithms
               X and Y Values Of The Training And Validation Sets
        Output: Prints The Prediction Results
    '''
    if algorithm == 'KNN':
        knnParametersTuning(train_X, train_Y, val_X, val_Y)
        model = KNeighborsClassifier(n_neighbors=3)
    if algorithm == 'SVM':
        parameters = {'kernel': ('linear', 'rbf'), 'C': [1, 5]}
        svr = svm.SVC(probability=False)
        model = GridSearchCV(svr, parameters, n_jobs=4, verbose=5)
        model.fit(train_X, train_Y)
        print("Tuned Hyperparameters :(best parameters) ", model.best_params_)
        print("Accuracy :", model.best_score_)
    if algorithm == 'LogReg':
        # Find Optimal Parameters
        #logRegParameterTuning(train_X, train_Y, val_X, val_Y)
        model = LogisticRegression(C=1000, penalty="l2")

    # Evaluate Final Model
    model.fit(train_X, train_Y)
    accuracyVal = model.score(val_X, val_Y)
    print("Algorithm: {} -> Validation Accuracy: {}".format(algorithm, accuracyVal))
    Y_score = model.predict(val_X)
    Y_classes = Y_score.argmax(axis=-1)

    loss = log_loss(val_Y, Y_score)
    #curve = auc(val_X, val_Y)
    precision = precision_score(val_Y, Y_score)
    recall = recall_score(val_Y, Y_score)
    f1 = f1_score(val_Y, Y_score)

    print("Algorithm: {} -> Validation Loss: {}".format(algorithm, loss))
    #print("Algorithm: {} -> Validation AUC: {}".format(algorithm, curve))
    print("Algorithm: {} -> Validation Precision: {}".format(algorithm, precision))
    print("Algorithm: {} -> Validation Recall: {}".format(algorithm, recall))
    print("Algorithm: {} -> Validation F1: {}".format(algorithm, f1))

    # tens = []
    # for i in range(len(Y_score_log)):
    #    tens.append(log_loss(y_test[i], Y_score_log[i]))

    #prob = model.predict_proba(test_X)
    accuracyTest = model.score(test_X, test_Y)

    return accuracyVal, accuracyTest


def create_modelNN(unit=8, activation='relu'):
    '''
        Creates A Fully Connected Neural Network Model

        Input: Receives Tuned Hyper-Parameters -> unit - Dimensionality Of The Output Space (int)
                                                  activation - Activation Function (string)
        Output: Created Keras Model
    '''
    model = Sequential()
    # Input Layer -> Rectified linear unit activation function
    model.add(Dense(137, input_dim=137, activation='relu'))
    # Hidden Layer -> Rectified linear unit activation function
    model.add(Dense(units=unit, activation='sigmoid'))
    # Output Layer - > Sigmoid Function
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='adam', metrics=['accuracy'])
    return model


def showComparison(acc):
    '''
        Illustrates The Results Difference Between The Trained Algorithms

        Input: acc - Array Of Float Values Corresponding To The Accuracy
        Output: Figure With Comparison (Saved To The Current Directory)
    '''
    plt.rcdefaults()
    tests = ['KNN', 'SVM', 'LogReg']
    y_pos = np.arange(len(tests))
    fig, ax = plt.subplots()
    ax.set_xlim(0, 1)
    ax.barh(y_pos, acc,  align='center')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(tests)
    ax.set_xlabel('Accuracy')
    ax.set_title('Accuracy Comparison Between Algorithms')
    plt.savefig('ML_Comparison.png', bbox_inches='tight')


def fullyConnectedNN(train_X, train_Y, val_X, val_Y, test_X, test_Y):
    '''
        Tuning of Hyper-Parameters, Training And Evaluation Of A Fully Connected Neural Network

        Input: X and Y Values Of The Training And Validation Sets
        Output: Prints The Prediction Results
    '''
    model = KerasClassifier(build_fn=create_modelNN, verbose=0)
    # define the grid search parameters
    unit = [70, 90, 110, 130, 150, 170, 190, 210, 230, 250, 270]
    param_grid = dict(unit=unit)
    grid = GridSearchCV(
        estimator=model, param_grid=param_grid, n_jobs=-1, cv=3)
    grid_result = grid.fit(train_X, train_Y)
    print("Best: %f using %s" %
          (grid_result.best_score_, grid_result.best_params_))

    #_, accuracy = grid.evaluate(val_X, val_Y)
    #print('Accuracy: %.2f' % (accuracy * 100))

    score_accuracy = grid.score(test_X, test_Y)
    print("Score Accuracy - Test : {}".format(score_accuracy))


def callModelTraining(train_X, train_Y, val_X, val_Y, test_X, test_Y):
    '''
        Creates Sequence Of Training And Evaluation For All Algorithms

        Input: X and Y Values Of The Training And Validation Sets
        Output: Figure With Result Comparison (Provided By The showComparison Function)
    '''
    result_accuracy = []
    result_acc = []
    #algorithms = ['KNN', 'SVM', 'LogReg']
    algorithms = ['KNN']
    for algo in algorithms:
        ra, rac = models(algo, train_X, train_Y, val_X, val_Y, test_X, test_Y)
        result_accuracy.append(ra)
        result_acc.append(rac)

    showComparison(result_acc)


if __name__ == "__main__":
    data = np.loadtxt('data/Shallue/separated/global_train.csv',
                      delimiter=',', skiprows=1)
    X = data[0:, 0:-1]  # Input
    Y = data[0:, -1]  # Labels
    imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    imp.fit(X)
    X = imp.transform(X)
    # Training = 80%, Validation = 10%, Test = 10%
    train_X, val_X, test_X = np.split(X, [int(.8*len(X)), int(0.9*len(X))])
    train_Y, val_Y, test_Y = np.split(Y, [int(.8*len(Y)), int(0.9*len(Y))])
    if np.isnan(train_X).any():
        print("nan")
    # ------------------------- NEW ------------------------------------------
    scaler_local = MinMaxScaler(feature_range=(0, 1))  # Scale Values
    train_X = scaler_local.fit_transform(train_X)
    val_X = scaler_local.transform(val_X)
    test_X = scaler_local.transform(test_X)

    callModelTraining(train_X, train_Y, val_X, val_Y, test_X, test_Y)
    #fullyConnectedNN(train_X, train_Y, val_X, val_Y, test_X, test_Y)
