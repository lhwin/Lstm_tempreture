import keras.backend as K
import keras

def max_percentage_error(y_true, y_pred):
    return K.max(K.abs((y_pred-y_true)/ K.clip(K.abs(y_true), K.epsilon(), None)))

def max_error(y_true, y_pred):
    return K.max(K.abs(y_true-y_pred), axis=-1)


class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.mae = []
        self.mse = []
        self.mpe = []
        self.max = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.mse.append(logs.get('mean_squared_error'))
        self.max.append(logs.get('max_error'))
        self.mpe.append(logs.get('max_percentage_error'))
