import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import itertools
from sklearn.model_selection import train_test_split
from keras.models import load_model
from sklearn.utils import shuffle

from utilities import chunkVisualization, recall_m, f1_m, precision_m, auc_roc


def chunks(data, n):
    """Yield successive n-sized chunks from lst"""
    for i in range(0, len(data), n):
        yield data[i:i + n]


if __name__ == "__main__":
    # Get Data
    data_global = np.loadtxt(
        '../data/Shallue/shallue_global.csv', delimiter=',', skiprows=1)
    data_global = shuffle(data_global)
    global_X = data_global[0:, 1:-1]  # Input
    global_Y = data_global[0:, -1]  # Labels

    # Scale Data
    scaler_global = MinMaxScaler(feature_range=(0, 1))  # Scale Values
    global_X = scaler_global.fit_transform(global_X)

    # Separate global Data
    X_train_global, X_test_global, y_train_global, y_test_global = train_test_split(
        global_X, global_Y, test_size=0.2, random_state=1)

    X_test_global = np.expand_dims(
        X_test_global, axis=2)    # Shape data

    # Get Model
    dependencies = {
        'f1_m': f1_m,
        'precision_m': precision_m,
        'recall_m': recall_m,
    }
    model = load_model('../models/CNN.h5', custom_objects=dependencies)
    score = model.evaluate(X_test_global, y_test_global, verbose=0)
    print("%s: %.2f%%" % (model.metrics_names[1], score[1]*100))

    prediction = model.predict_classes(X_test_global)
    for i in range(len(prediction)):
        # if (prediction[i] == 1):
        #    print(i)
        print("X=%s, Predicted=%s" % (y_test_global[i], prediction[i]))
    #print("X=%s, Predicted=%s" % (y_test_global[880], prediction[880]))
    # chuncked_data = list(chunks(global_X[0], 50))
    # combinations = list(itertools.combinations(chuncked_data, 3))

    # chunkVisualization(global_X[0], 50)

    plt.show()
