import numpy as np
from pycaret.anomaly import *
from sklearn.model_selection import train_test_split
import pandas as pd


if __name__ == "__main__":
    data_global = pd.read_csv(
        'data/Shallue/separated/global_test.csv', delimiter=',')

    global_X = data_global.iloc[0:, 0:-1].copy()  # Input
    global_Y = data_global.iloc[:, -1].copy()  # Labels

    X_train_global, X_test_global, y_train_global, y_test_global = train_test_split(
        global_X, global_Y, test_size=0.1, random_state=1)

    exp_ano101 = setup(X_train_global, normalize=True,
                       session_id=123)

    iforest = create_model('iforest')
    iforest_results = assign_model(iforest)
    plot_model(iforest)

    unseen_predictions = predict_model(iforest, data=X_test_global)
    data_predictions = predict_model(iforest, data=X_train_global)
    data_predictions.head()
