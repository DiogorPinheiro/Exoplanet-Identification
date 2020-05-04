from kerastuner import HyperModel
import tensorflow as tf
from tensorflow import keras
from keras import optimizers
from sklearn.model_selection import train_test_split
from keras.layers import Dense, Input, concatenate, Flatten, Dropout, PReLU, BatchNormalization, Activation, GaussianNoise, MaxPooling1D, LSTM
from keras.layers.convolutional import Conv1D
from training import f1_m, precision_m, recall_m, mainEvaluate, auc_roc


class CNNTrial2(HyperModel):

    def __init__(self, num_classes):
        self.num_classes = num_classes

    def build(self, hp):

        # Specify model
        model = keras.Sequential()

        # Range of models to build
        for i in range(hp.Int('num_layers', 2, 20)):

            model.add(keras.layers.Dense(units=hp.Int('units_' + str(i),
                                                      min_value=32,
                                                      max_value=512,
                                                      step=32),
                                         activation='relu'))

        # Output layer
        model.add(keras.layers.Dense(self.num_classes, activation='sigmoid'))

        # Compile the constructed model and return it
        model.compile(
            optimizer=keras.optimizers.Adam(
                hp.Choice('learning_rate',
                          values=[1e-2, 1e-3, 1e-4])),
            loss='binary_crossentropy',
            metrics=['accuracy'])

        return model


class CNNTrial(HyperModel):
    def __init__(self, input_shape, num_classes):
        self.input_shape = input_shape
        self.num_classes = num_classes

    def build_model(self, hp):
        inputs = tf.keras.Input(shape=self.input_shape)
        x = inputs
        for i in range(hp.Range('conv_blocks', 0, 5, default=3)):
            filters = hp.Int('filters_' + str(i), 32, 256, step=32)
            x = tf.keras.layers.Conv1D(
                filters, kernel_size=hp.Int(
                    'kernel_size_'+str(i), 1, 5, default=3), padding='same')(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.ReLU()(x)
            if hp.Choice('pooling_' + str(i), ['avg', 'max']) == 'max':
                x = tf.keras.layers.MaxPool1D()(x)
            else:
                x = tf.keras.layers.AvgPool1D()(x)
        for j in range(hp.Range('dense_blocks', 0, 5, default=3)):
            x = tf.keras.layers.Dense(
                hp.Int('hidden_size_'+str(j), 30, 100, step=10, default=50),
                activation='relu')(x)
            x = tf.keras.layers.Dropout(
                hp.Float('dropout_'+str(j), 0, 0.5, step=0.1, default=0.5))(x)
        outputs = tf.keras.layers.Dense(
            self.num_classes, activation='sigmoid')(x)

        model = tf.keras.Model(inputs, outputs)
        model.compile(optimizer=tf.keras.optimizers.SGD(hp.Float('lr', 0, 0.1, step=0.001, default=0.01), hp.Float('momentum', 0, 0.5, step=0.01, default=0.1)),
                      loss='binary_crossentropy',
                      metrics=['accuracy', f1_m, precision_m, recall_m, tf.keras.metrics.AUC()])
        return model


class CNNHyperModel(HyperModel):
    def __init__(self, input_shape, num_classes):
        self.input_shape = input_shape
        self.num_classes = num_classes

    def build(self, hp):
        model = keras.Sequential()
        model.add(
            Conv1D(
                filters=hp.Choice("filters", [
                                  8, 16, 32, 64, 128, 254], default=64),
                kernel_size=hp.Int(
                    'kernel_size', 1, 5, default=3),
                activation="relu",
                input_shape=self.input_shape,
            )
        )
        for i in range(hp.Range('conv_blocks', 1, 7, default=3)):
            model.add(Conv1D(filters=hp.Choice("filters_"+str(i), [
                8, 16, 32, 64, 128, 254], default=64), activation="relu", kernel_size=hp.Int(
                    'kernel_size_'+str(i), 1, 5, default=3)))
            model.add(MaxPooling1D(pool_size=hp.Int(
                'pool_size_'+str(i), 1, 5, default=3)))
            model.add(
                Dropout(
                    rate=hp.Float(
                        "dropout_"+str(i), min_value=0.0, max_value=0.5, default=0.25, step=0.05,
                    )
                )
            )

        model.add(Flatten())
        for f in range(hp.Range('dense_blocks', 0, 5, default=2)):
            model.add(
                Dense(
                    units=hp.Int(
                        "units_"+str(f), min_value=32, max_value=512, step=32, default=128
                    ),
                    activation=hp.Choice(
                        "dense_activation_"+str(f),
                        values=["relu", "tanh", "sigmoid"],
                        default="relu",
                    ),
                )
            )
            model.add(
                Dropout(
                    rate=hp.Float(
                        "dropout_"+str(f), min_value=0.0, max_value=0.5, default=0.25, step=0.05
                    )
                )
            )
        model.add(Dense(self.num_classes, activation="sigmoid"))

        model.compile(
            optimizer=keras.optimizers.Adam(
                hp.Float(
                    "learning_rate",
                    min_value=1e-4,
                    max_value=1e-2,
                    default=1e-3,
                )
            ),
            loss="binary_crossentropy",
            metrics=['accuracy', f1_m, precision_m,
                     recall_m, tf.keras.metrics.AUC()],
        )
        return model


class LSTMHyperModel(HyperModel):
    def __init__(self, input_shape, num_classes):
        self.input_shape = input_shape
        self.num_classes = num_classes

    def build(self, hp):
        inputLayer = Input(shape=self.input_shape)

        model = LSTM(units=hp.Int('LSTM_units1', 1, 15, default=5), return_sequences=True,
                     unit_forget_bias=True, bias_initializer='zeros')(inputLayer)
        model = PReLU()(model)
        model = Dropout(hp.Float("dropout1", min_value=0.0,
                                 max_value=0.5, default=0.2, step=0.05))(model)

        for i in range(hp.Range('lstm_blocks', 0, 7, default=3)):
            model = LSTM(units=hp.Int('LSTM_units_'+str(i), 1, 15, default=5), unit_forget_bias=True,
                         bias_initializer='zeros', return_sequences=True)(model)
            model = PReLU()(model)
            model = Dropout(hp.Float("dropout_"+str(i), min_value=0.0,
                                     max_value=0.5, default=0.2, step=0.05))(model)

        model = Flatten()(model)

        for f in range(hp.Range('dense_blocks', 0, 7, default=2)):
            model = Dense(units=hp.Int("units_"+str(f), min_value=32,
                                       max_value=512, step=32, default=128))(model)
            model = PReLU()(model)
            model = Dropout(hp.Float("dropout_"+str(i), min_value=0.0,
                                     max_value=0.5, default=0.2, step=0.05))(model)

        model.add(Dense(self.num_classes, activation="sigmoid"))

        model.compile(
            optimizer=keras.optimizers.Adam(
                hp.Float(
                    "learning_rate",
                    min_value=1e-4,
                    max_value=1e-2,
                    default=1e-3,
                )
            ),
            loss="binary_crossentropy",
            metrics=['accuracy', f1_m, precision_m,
                     recall_m, tf.keras.metrics.AUC()],
        )
        return model


class FNNHyperModel(HyperModel):
    def __init__(self, input_shape, num_classes):
        self.input_shape = input_shape
        self.num_classes = num_classes

    def build(self, hp):
        inputLayer = Input(shape=self.input_shape)

        model = Dense(units=hp.Int("units", min_value=32, max_value=512, step=32, default=128),
                      activation=hp.Choice("dense_activation",
                                           values=["relu", "tanh", "sigmoid"],
                                           default="relu",
                                           ),)(inputLayer)

        for f in range(hp.Range('dense_blocks', 0, 5, default=2)):
            model = Dense(units=hp.Int("units_f"+str(f), min_value=32, max_value=512, step=32, default=128),
                          activation=hp.Choice("dense_activation",
                                               values=[
                                                   "relu", "tanh", "sigmoid"],
                                               default="relu",
                                               ),)(model)

        model.add(Dense(self.num_classes, activation="sigmoid"))

        model.compile(
            optimizer=keras.optimizers.Adam(
                hp.Float(
                    "learning_rate",
                    min_value=1e-4,
                    max_value=1e-2,
                    default=1e-3,
                )
            ),
            loss="binary_crossentropy",
            metrics=['accuracy', f1_m, precision_m,
                     recall_m, tf.keras.metrics.AUC()],
        )
        return model


class DualCNNHyperModel(HyperModel):
    def __init__(self, input_shape_local, input_shape_global, num_classes):
        self.input_shape_local = input_shape_local
        self.input_shape_global = input_shape_global
        self.num_classes = num_classes

    def build(self, hp):
        inputLayer_local = Input(shape=self.input_shape_local)
        inputLayer_global = Input(shape=self.input_shape_global)
