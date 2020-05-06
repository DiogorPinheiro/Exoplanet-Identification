from kerastuner import HyperModel
from kerastuner.engine.hyperparameters import HyperParameters
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
        for i in range(hp.Int('conv_blocks', 0, 5, default=3)):
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
        for j in range(hp.Int('dense_blocks', 0, 5, default=3)):
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
    def __init__(self, input_shape, num_classes):
        self.input_shape = input_shape
        self.num_classes = num_classes

    def build(self, hp):
        inputLayer = tf.keras.layers.Input(shape=self.input_shape)

        model = tf.keras.layers.LSTM(units=hp.Int('LSTM1', 1, 15, default=5), return_sequences=True,
                                     unit_forget_bias=True, bias_initializer='zeros')(inputLayer)
        model = tf.keras.layers.PReLU()(model)
        model = tf.keras.layers.Dropout(hp.Float("dropout1", min_value=0.0,
                                                 max_value=0.5, default=0.2, step=0.05))(model)

        for i in range(hp.Int('lstm_blocks', 0, 7, default=3)):
            model = tf.keras.layers.LSTM(units=hp.Int('LSTM_units_'+str(i), 1, 15, default=5), unit_forget_bias=True,
                                         bias_initializer='zeros', return_sequences=True)(model)
            model = tf.keras.layers.PReLU()(model)
            model = tf.keras.layers.Dropout(hp.Float("dropout_"+str(i), min_value=0.0,
                                                     max_value=0.5, default=0.2, step=0.05))(model)

        model = tf.keras.layers.Flatten()(model)

        for f in range(hp.Int('dense_blocks', 0, 7, default=2)):
            model = tf.keras.layers.Dense(units=hp.Int("units_"+str(f), min_value=32,
                                                       max_value=512, step=32, default=128))(model)
            model = tf.keras.layers.PReLU()(model)
            model = tf.keras.layers.Dropout(hp.Float("dropout_"+str(i), min_value=0.0,
                                                     max_value=0.5, default=0.2, step=0.05))(model)

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


class FNNHyperModel(HyperModel):
    def __init__(self, input_shape, num_classes):
        self.input_shape = input_shape
        self.num_classes = num_classes

    def build(self, hp):
        inputLayer = Input(shape=self.input_shape)

        model = tf.keras.layers.Dense(units=hp.Int("units", min_value=32, max_value=512, step=32, default=128),
                                      activation=hp.Choice("dense_activation",
                                                           values=[
                                                               "relu", "tanh", "sigmoid"],
                                                           default="relu",
                                                           ),)(inputLayer)

        for f in range(hp.Int('dense_blocks', 0, 5, default=2)):
            model = tf.keras.layers.Dense(units=hp.Int("units_f"+str(f), min_value=32, max_value=512, step=32, default=128),
                                          activation=hp.Choice("dense_activation",
                                                               values=[
                                                                   "relu", "tanh", "sigmoid"],
                                                               default="relu",
                                                               ),)(model)

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


class DualCNNHyperModel(HyperModel):
    def __init__(self, input_shape_local, input_shape_global, num_classes):
        self.input_shape_local = input_shape_local
        self.input_shape_global = input_shape_global
        self.num_classes = num_classes

    def build(self, hp):
        inputLayer_local = Input(shape=self.input_shape_local)
        inputLayer_global = Input(shape=self.input_shape_global)

        conv_local = tf.keras.layers.Conv1D(hp.Int("Conv_unis1", min_value=8, max_value=254, step=32, default=128), kernel_size=hp.Int("kernel1", min_value=0, max_value=7, step=1, default=4),
                                            strides=hp.Int("strides1", min_value=1, max_value=4, step=1, default=2), padding='same', dilation_rate=1,
                                            activation='relu', kernel_initializer='he_normal')
        conv_global = tf.keras.layers.Conv1D(hp.Int("Conv_unis2", min_value=8, max_value=254, step=32, default=128), kernel_size=hp.Int("kernel2", min_value=0, max_value=7, step=1, default=4),
                                             strides=hp.Int("strides2", min_value=1, max_value=4, step=1, default=2), padding='same', dilation_rate=1,
                                             activation='relu', kernel_initializer='he_normal')

        model1 = conv_global(inputLayer_global)  # Disjoint Conv Layer
        model1 = tf.keras.layers.MaxPooling1D(pool_size=hp.Int(
            "pool1", min_value=1, max_value=4, step=1, default=2), strides=hp.Int("stri1", min_value=1, max_value=4, step=1, default=2), padding='valid')(model1)
        model1 = tf.keras.layers.Dropout(hp.Float("dropout1", min_value=0.0,
                                                  max_value=0.5, default=0.2, step=0.05))(model1)

        for f in range(hp.Int('ConvGlobal_blocks', 0, 10, default=2)):
            model1 = tf.keras.layers.Conv1D(hp.Int("Conv_gl_"+str(f), min_value=8, max_value=254, step=32, default=128),  kernel_size=hp.Int("kern_gl_"+str(f), min_value=0, max_value=7, step=1, default=4),
                                            strides=hp.Int("stri_gl_"+str(f), min_value=1, max_value=4, step=1, default=2), padding='same',
                                            dilation_rate=1, activation='relu')(model1)
            model1 = tf.keras.layers.MaxPooling1D(pool_size=hp.Int("pool_gl_"+str(f), min_value=1, max_value=4, step=1, default=2), strides=hp.Int("max_str_gl_"+str(f), min_value=1, max_value=4, step=1, default=2),
                                                  padding='valid')(model1)
            model1 = tf.keras.layers.Dropout(hp.Float("dropout_gl_"+str(f), min_value=0.0,
                                                      max_value=0.5, default=0.2, step=0.05))(model1)
        model1 = tf.keras.layers.Flatten()(model1)

        model2 = conv_local(inputLayer_local)  # Disjoint Conv Layer
        model2 = tf.keras.layers.MaxPooling1D(pool_size=hp.Int(
            "pool2", min_value=1, max_value=4, step=1, default=2), strides=hp.Int("stri2", min_value=1, max_value=4, step=1, default=2), padding='valid')(model2)
        model2 = tf.keras.layers.Dropout(hp.Float("dropout2", min_value=0.0,
                                                  max_value=0.5, default=0.2, step=0.05))(model2)

        for i in range(hp.Int('ConvLocal_blocks', 0, 10, default=2)):
            model2 = tf.keras.layers.Conv1D(hp.Int("Conv_lc_"+str(i), min_value=8, max_value=254, step=32, default=128),  kernel_size=hp.Int("kern_lc_"+str(i), min_value=0, max_value=7, step=1, default=4),
                                            strides=hp.Int("stri_lc_"+str(i), min_value=1, max_value=4, step=1, default=2), padding='same',
                                            dilation_rate=1, activation='relu')(model2)
            model2 = tf.keras.layers.MaxPooling1D(pool_size=hp.Int("pool_lc_"+str(i), min_value=1, max_value=4, step=1, default=2), strides=hp.Int("max_str_lc_"+str(i), min_value=1, max_value=4, step=1, default=2),
                                                  padding='valid')(model2)
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

        model.add(tf.keras.layers.Dense(
            self.num_classes, activation="sigmoid"))

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
