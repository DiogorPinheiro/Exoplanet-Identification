from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import GridSearchCV
import  numpy as np
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerLine2D

def knnParametersTuning(train_X,train_Y,val_X,val_Y):
    k = list(range(1, 30))
    train_results = []
    test_results = []
    for i in k:
        model = KNeighborsClassifier(n_neighbors=i)
        model.fit(train_X, train_Y)
        train_pred = model.predict(train_X)

        false_positive_rate, true_positive_rate, thresholds = roc_curve(train_Y, train_pred)    # Using Area Under Curve As Evaluation Metric
        roc_auc = auc(false_positive_rate, true_positive_rate)

        train_results.append(roc_auc)

        y_pred = model.predict(val_X)
        false_positive_rate, true_positive_rate, thresholds = roc_curve(val_Y, y_pred)
        roc_auc = auc(false_positive_rate, true_positive_rate)

        test_results.append(roc_auc)
    print(test_results)

    line1, = plt.plot(k, train_results, 'b', label="Train AUC")
    line2, = plt.plot(k, test_results, 'r', label="Validation AUC")
    plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
    plt.ylabel('AUC Score')
    plt.xlabel('Power Parameter')
    plt.savefig('knnScores.png', bbox_inches='tight')

def logRegParameterTuning(train_X,train_Y,val_X,val_Y):
    grid = {"C": np.logspace(-3, 3, 7), "penalty": ["l1", "l2"]}  # l1 lasso l2 ridge
    logreg = LogisticRegression()
    logreg_cv = GridSearchCV(logreg, grid, cv=10)
    logreg_cv.fit(train_X, train_Y)
    print("Tuned Hyperparameters :(best parameters) ", logreg_cv.best_params_)
    print("Accuracy :", logreg_cv.best_score_)
    #smaller C specify stronger regularization.

def svmParameterTuning(train_X,train_Y,val_X,val_Y):
    parameters = {'kernel': ('linear', 'rbf'), 'C': [1, 10]}
    svr = svm.SVC(probability=False)
    model = GridSearchCV(svr, parameters,n_jobs=4)
    model.fit(train_X, train_Y)
    print("Tuned Hyperparameters :(best parameters) ", model.best_params_)
    print("Accuracy :", model.best_score_)

def models(algorithm, train_X,train_Y,val_X,val_Y,test_X, test_Y):
    if algorithm == 'KNN':
        #knnParametersTuning(train_X, train_Y, val_X, val_Y)
        model = KNeighborsClassifier(n_neighbors=3)
    if algorithm == 'SVM':
        parameters = {'kernel': ('linear', 'rbf'), 'C': [1, 10]}
        svr = svm.SVC(probability=False)
        model = GridSearchCV(svr, parameters,n_jobs=4)
        model.fit(train_X, train_Y)
        print("Tuned Hyperparameters :(best parameters) ", model.best_params_)
        print("Accuracy :", model.best_score_)
    if algorithm == 'LogReg':
        logRegParameterTuning(train_X,train_Y,val_X,val_Y)  # Find Optimal Parameters
        model = LogisticRegression(C=0.001,penalty="l2")

    # Evaluate Final Model
    model.fit(train_X,train_Y)
    accuracyVal = model.score(val_X, val_Y)
    print("Algorithm: {} -> Validation Accuracy: {}".format(algorithm,accuracyVal))

    #prob = model.predict_proba(test_X)
    accuracyTest = model.score(test_X,test_Y)

    return  accuracyVal, accuracyTest

def showComparison(acc):
    plt.rcdefaults()
    tests = ['KNN', 'SVM', 'LogReg']
    y_pos = np.arange(len(tests))
    fig, ax = plt.subplots()
    ax.set_xlim(0,1)
    ax.barh(y_pos, acc,  align='center')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(tests)
    ax.set_xlabel('Accuracy')
    ax.set_title('Accuracy Comparison Between Algorithms')
    plt.savefig('ML_Comparison.png', bbox_inches='tight')

def main():
    data=np.loadtxt('dataset.csv',delimiter=',',skiprows=1)
    X=data[1:,0:-1] # Input
    Y= data[1:,-1]  # Labels
    train_X,val_X,test_X=np.split(X,[int(.8*len(X)),int(0.9*len(X))])  # Training = 80%, Validation = 10%, Test = 10%
    train_Y,val_Y,test_Y=np.split(Y,[int(.8*len(Y)),int(0.9*len(Y))])

    result_accuracy = []
    result_acc = []
    algorithms = ['KNN','SVM','LogReg']
    #algorithms = ['KNN','LogReg']
    for algo in algorithms:
        ra,rac = models(algo,train_X,train_Y,val_X,val_Y,test_X,test_Y)
        result_accuracy.append(ra)
        result_acc.append(rac)

    showComparison(result_acc)

main()