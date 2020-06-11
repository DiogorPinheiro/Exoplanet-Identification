from kerastuner import HyperModel
from kerastuner.engine.hyperparameters import HyperParameters
import tensorflow as tf
from tensorflow import keras
from keras import optimizers
from keras.models import Model
from keras.layers import Dense, Input, concatenate, Flatten, Dropout, PReLU, BatchNormalization, Activation, GaussianNoise, MaxPooling1D, LSTM
from keras.layers.convolutional import Conv1D

from utilities import f1_m, precision_m, recall_m


class CNNHyperModel(HyperModel):
    '''
        Find the best CNN model.

        @param HyperModel (object): Hypermodel subclass

    '''

    def __init__(self, input_shape, num_classes):
        self.input_shape = input_shape
        self.num_classes = num_classes

    def build(self, hp):
        model = keras.Sequential()
        model.add(
            tf.keras.layers.Conv1D(
                filters=hp.Choice("filter1", [
                                  8, 16, 32, 64, 128, 254], default=64),
                kernel_size=hp.Choice(
                    'kernel1',  [
                        3, 5, 7, 9], default=5),
                activation="relu",
                input_shape=self.input_shape,
            )
        )
        model.add(tf.keras.layers.MaxPooling1D(pool_size=hp.Choice(
            'pool_s', [
                2, 3, 4], default=3)))
        model.add(
            tf.keras.layers.Dropout(
                rate=hp.Float(
                    "drop", min_value=0.0, max_value=0.5, default=0.25, step=0.05,
                )
            )
        )
        for i in range(hp.Int('conv_blocks', 0, 3, default=3)):
            model.add(tf.keras.layers.Conv1D(filters=hp.Choice("filters_"+str(i), [
                8, 16, 32, 64, 128, 254], default=64), activation="relu", kernel_size=hp.Choice(
                    'kernel_size_'+str(i), [
                        3, 5, 7, 9], default=5)))
            model.add(tf.keras.layers.MaxPooling1D(pool_size=hp.Choice(
                'pool_size_'+str(i),  [
                    2, 3, 4], default=3)))
            model.add(
                tf.keras.layers.Dropout(
                    rate=hp.Float(
                        "dropout_"+str(i), min_value=0.0, max_value=0.5, default=0.25, step=0.05,
                    )
                )
            )

        model.add(tf.keras.layers.Flatten())
        for f in range(hp.Int('dense_blocks', 0, 5, default=2)):
            model.add(
                tf.keras.layers.Dense(
                    units=hp.Choice(
                        "units_"+str(f),  [
                            32, 64, 128, 254], default=64
                    ),
                    activation=hp.Choice(
                        "dense_activation_"+str(f),
                        values=["relu", "tanh", "sigmoid"],
                        default="relu",
                    ),
                )
            )
            model.add(
                tf.keras.layers.Dropout(
                    rate=hp.Float(
                        "dropout2_"+str(f), min_value=0.0, max_value=0.5, default=0.25, step=0.05
                    )
                )
            )
        model.add(tf.keras.layers.Dense(
            self.num_classes, activation="sigmoid"))

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

    def get_config(self):
        return {
            'input_shape': self.input_shape,
            'num_classes': self.num_classes
        }


class LSTMHyperModel(HyperModel):
    '''
        Find the best LSTM model.

        @param HyperModel (object): Hypermodel subclass

    '''

    def __init__(self, input_shape, num_classes):
        self.input_shape = input_shape
        self.num_classes = num_classes

    def build(self, hp):
        inputLayer = tf.keras.layers.Input(shape=self.input_shape)

        model = tf.keras.layers.LSTM(units=hp.Choice('LSTM1', [
            2, 5, 10, 15], default=5), return_sequences=True,
            unit_forget_bias=True, bias_initializer='zeros')(inputLayer)
        model = tf.keras.layers.PReLU()(model)
        model = tf.keras.layers.Dropout(hp.Float("dropout1", min_value=0.0,
                                                 max_value=0.5, default=0.2, step=0.05))(model)

        for i in range(hp.Int('lstm_blocks', 0, 3, default=3)):
            model = tf.keras.layers.LSTM(units=hp.Choice('LSTM_units_'+str(i), [
                2, 5, 10, 15], default=5), unit_forget_bias=True,
                bias_initializer='zeros', return_sequences=True)(model)
            model = tf.keras.layers.PReLU()(model)
            model = tf.keras.layers.Dropout(hp.Float("dropout_lstm_"+str(i), min_value=0.0,
                                                     max_value=0.5, default=0.2, step=0.05))(model)

        model = tf.keras.layers.Flatten()(model)

        for f in range(hp.Int('dense_blocks', 0, 7, default=2)):
            model = tf.keras.layers.Dense(units=hp.Choice("dense_units_"+str(f), [
                32, 64, 128], default=64))(model)
            model = tf.keras.layers.PReLU()(model)
            model = tf.keras.layers.Dropout(hp.Float("dropout_dense_"+str(f), min_value=0.0,
                                                     max_value=0.5, default=0.2, step=0.05))(model)

        out = tf.keras.layers.Dense(
            self.num_classes, activation="sigmoid")(model)
        model = tf.keras.models.Model(inputs=inputLayer, outputs=out)

        model.compile(
            optimizer=tf.keras.optimizers.Adam(
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

    def get_config(self):
        return {
            'input_shape': self.input_shape,
            'num_classes': self.num_classes
        }


class FNNHyperModel(HyperModel):
    '''
        Find the best FNN model.

        @param HyperModel (object): Hypermodel subclass

    '''

    def __init__(self, input_shape, num_classes):
        self.num_classes = num_classes
        self.input_shape = input_shape

    def build(self, hp):
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Flatten(input_shape=self.input_shape))
        model.add(tf.keras.layers.Dense(units=hp.Int("units", min_value=32, max_value=512, step=32, default=128),
                                        activation='relu'))

        for f in range(hp.Int('dense_blocks', 1, 7, default=2)):
            model.add(tf.keras.layers.Dense(units=hp.Int("units_f"+str(f), min_value=32, max_value=512, step=32, default=128),
                                            activation='relu'))

        model.add(tf.keras.layers.Dense(
            self.num_classes, activation='sigmoid'))

        model.compile(
            optimizer=tf.keras.optimizers.Adam(
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

    def get_config(self):
        return {
            'input_shape': self.input_shape,
            'num_classes': self.num_classes
        }


class DualCNNHyperModel(HyperModel):
    '''
        Find the best Dual CNN model.

        @param HyperModel (object): Hypermodel subclass

    '''

    def __init__(self, input_shape_local, input_shape_global, num_classes):
        self.input_shape_local = input_shape_local
        self.input_shape_global = input_shape_global
        self.num_classes = num_classes

    def build(self, hp):
        inputLayer_local = tf.keras.Input(shape=self.input_shape_local)
        inputLayer_global = tf.keras.Input(shape=self.input_shape_global)

        conv_local = tf.keras.layers.Conv1D(hp.Int("Conv_unis1", min_value=8, max_value=254, step=32, default=128), kernel_size=hp.Int("kernel1", min_value=0, max_value=7, step=1, default=4),
                                            strides=hp.Int("strides1", min_value=1, max_value=4, step=1, default=2), padding='same', dilation_rate=1,
                                            activation='relu', kernel_initializer='he_normal')
        conv_global = tf.keras.layers.Conv1D(hp.Int("Conv_unis2", min_value=8, max_value=254, step=32, default=128), kernel_size=hp.Int("kernel2", min_value=0, max_value=7, step=1, default=4),
                                             strides=hp.Int("strides2", min_value=1, max_value=4, step=1, default=2), padding='same', dilation_rate=1,
                                             activation='relu', kernel_initializer='he_normal')

        model1 = conv_global(inputLayer_global)  # Disjoint Conv Layer
        model1 = tf.keras.layers.MaxPooling1D(pool_size=hp.Int(
            "pool1", min_value=1, max_value=4, step=1, default=2), strides=hp.Int("stri1", min_value=1, max_value=4, step=1, default=2), padding='same')(model1)
        model1 = tf.keras.layers.Dropout(hp.Float("dropout1", min_value=0.0,
                                                  max_value=0.5, default=0.2, step=0.05))(model1)

        for f in range(hp.Int('ConvGlobal_blocks', 0, 10, default=2)):
            model1 = tf.keras.layers.Conv1D(hp.Int("Conv_gl_"+str(f), min_value=8, max_value=254, step=32, default=128),  kernel_size=hp.Int("kern_gl_"+str(f), min_value=0, max_value=7, step=1, default=4),
                                            strides=hp.Int("stri_gl_"+str(f), min_value=1, max_value=4, step=1, default=2), padding='same',
                                            dilation_rate=1, activation='relu')(model1)
            model1 = tf.keras.layers.MaxPooling1D(pool_size=hp.Int("pool_gl_"+str(f), min_value=1, max_value=4, step=1, default=2), strides=hp.Int("max_str_gl_"+str(f), min_value=1, max_value=4, step=1, default=2),
                                                  padding='same')(model1)
            model1 = tf.keras.layers.Dropout(hp.Float("dropout_gl_"+str(f), min_value=0.0,
                                                      max_value=0.5, default=0.2, step=0.05))(model1)
        model1 = tf.keras.layers.Flatten()(model1)

        model2 = conv_local(inputLayer_local)  # Disjoint Conv Layer
        model2 = tf.keras.layers.MaxPooling1D(pool_size=hp.Int(
            "pool2", min_value=1, max_value=4, step=1, default=2), strides=hp.Int("stri2", min_value=1, max_value=4, step=1, default=2), padding='same')(model2)
        model2 = tf.keras.layers.Dropout(hp.Float("dropout2", min_value=0.0,
                                                  max_value=0.5, default=0.2, step=0.05))(model2)

        for i in range(hp.Int('ConvLocal_blocks', 0, 10, default=2)):
            model2 = tf.keras.layers.Conv1D(hp.Int("Conv_lc_"+str(i), min_value=8, max_value=254, step=32, default=128),  kernel_size=hp.Int("kern_lc_"+str(i), min_value=0, max_value=7, step=1, default=4),
                                            strides=hp.Int("stri_lc_"+str(i), min_value=1, max_value=4, step=1, default=2), padding='same',
                                            dilation_rate=1, activation='relu')(model2)
            model2 = tf.keras.layers.MaxPooling1D(pool_size=hp.Int("pool_lc_"+str(i), min_value=1, max_value=4, step=1, default=2), strides=hp.Int("max_str_lc_"+str(i), min_value=1, max_value=4, step=1, default=2),
                                                  padding='same')(model2)
            model2 = tf.keras.layers.Dropout(hp.Float("dropout_lc_"+str(i), min_value=0.0,
                                                      max_value=0.5, default=0.2, step=0.05))(model2)
        model2 = tf.keras.layers.Flatten()(model2)

        concatLayerQ = keras.layers.concatenate([model1, model2], axis=1)

        model = tf.keras.layers.Dense(units=hp.Int("uni1", min_value=32,
                                                   max_value=512, step=32, default=128))(concatLayerQ)
        model = tf.keras.layers.BatchNormalization()(model)
        model = tf.keras.layers.Dropout(hp.Float("drop1", min_value=0.0,
                                                 max_value=0.5, default=0.2, step=0.05))(model)

        for j in range(hp.Int('dense_blocks', 0, 7, default=2)):
            model = tf.keras.layers.Dense(units=hp.Int("units_ds_"+str(j), min_value=32,
                                                       max_value=512, step=32, default=128))(model)
            model = tf.keras.layers.BatchNormalization()(model)
            model = tf.keras.layers.Dropout(hp.Float("dropout_ds_"+str(j), min_value=0.0,
                                                     max_value=0.5, default=0.2, step=0.05))(model)

        out = tf.keras.layers.Dense(
            self.num_classes, activation="sigmoid")(model)
        model = tf.keras.models.Model(inputs=[inputLayer_local,
                                              inputLayer_global], outputs=out)

        model.compile(
            optimizer=keras.optimizers.Adam(
                hp.Float(
                    "learning_rate",
                    min_value=1e-4,
                    max_value=1e-2,
                    default=1e-3,
                ), beta_1=hp.Float(
                    "beta_1",
                    min_value=0.7,
                    max_value=0.95,
                    default=1e-3,
                ),
                beta_2=0.999, amsgrad=False
            ),
            loss="binary_crossentropy",
            metrics=['accuracy', f1_m, precision_m,
                     recall_m, tf.keras.metrics.AUC()],
        )
        return model

    def get_config(self):
        return {
            'input_shape': self.input_shape,
            'num_classes': self.num_classes
        }
