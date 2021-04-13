from keras.models import Model
from keras.optimizers import Adam, RMSprop
from keras.layers import LSTM, Input, Dense, Flatten,SimpleRNN, Lambda, BatchNormalization
from keras.optimizers import Adam
from Data_process import concat_data
from Evluate import max_error, max_percentage_error, LossHistory
from keras import losses
import keras.backend as K
from Cosine_decay import WarmUpCosineDecayScheduler,SnapshotEnsemble, LearningRateScheduler
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

class Models(object):
    def __init__(self, batch_size, input):
        self.batch_size = batch_size
        self.opt = Adam(1e-4)
        self.losses = ['binary_crossentropy', 'binary_crossentropy']
        self.lstm_model = self.con_lstm(input)
        self.bp = self.con_bp(input)
        self.rnn = self.con_rnn(input)

    def con_lstm(self, input):
        x = Lambda(lambda x:K.expand_dims(x, axis=-1))(input)
        x = LSTM(units=20, input_shape=(13, ), return_sequences=False)(x)

        x = Dense(100, activation='relu')(x)
        x = Dense(50, activation='relu')(x)

        out = Dense(7, activation="sigmoid")(x)

        LSTM_model = Model(input, out)

        LSTM_model.summary()

        return LSTM_model

    def con_bp(self, input):
        x = Dense(10, activation='relu')(input)
        x = Dense(10, activation='relu')(x)
        out = Dense(7, activation="sigmoid")(x)

        BP_model = Model(input, out)
        BP_model.summary()

        return BP_model

    def con_rnn(self, input):
        x = Lambda(lambda x: K.expand_dims(x, axis=-1))(input)
        x = SimpleRNN(25, return_sequences=False, input_shape=(13, ), unroll=True)(x)
        x = Dense(10, activation='relu')(x)
        out = Dense(7, activation="sigmoid")(x)

        RNN_model = Model(input, out)
        RNN_model.summary()

        return RNN_model

    def compile_model(self, model):
        model.compile(loss="mae", optimizer=self.opt, metrics=['mae', 'mse', max_error, max_percentage_error])

    def write_history(self, model_name, history):
        with open("./log/"+model_name.split(".")[0]+"loss_log.txt", "a", encoding="utf-8") as f:
            for ele in history.losses:
                f.write(str(ele)+"\n")
        with open("./log/" + model_name.split(".")[0] + "mse_log.txt", "a", encoding="utf-8") as f:
            for ele in history.mae:
                f.write(str(ele) + "\n")
        with open("./log/"+model_name.split(".")[0]+"mse_log.txt", "a", encoding="utf-8") as f:
            for ele in history.mse:
                f.write(str(ele)+"\n")
        with open("./log/"+model_name.split(".")[0]+"mpe_log.txt", "a", encoding="utf-8") as f:
            for ele in history.mpe:
                f.write(str(ele)+"\n")
        with open("./log/"+model_name.split(".")[0]+"max_log.txt", "a", encoding="utf-8") as f:
            for ele in history.mpe:
                f.write(str(ele)+"\n")

    def train(self, model, X, Y, model_name, epochs):
        self.compile_model(model)
        epochs = int(epochs)
        history = LossHistory()
        warmup_epoch = 20
        total_steps = int(epochs * X.shape[0] / self.batch_size)
        warmup_steps = int(warmup_epoch * X.shape[0] / self.batch_size)
        Warm_up_lr = WarmUpCosineDecayScheduler(learning_rate_base=0.01,
                                               total_steps=total_steps,
                                               warmup_learning_rate=4e-06,
                                               warmup_steps=warmup_steps,
                                               hold_base_rate_steps=5,

                                               )

        n_epochs = epochs

        n_cycles = n_epochs / 5

        ca = SnapshotEnsemble(n_epochs, n_cycles, 0.01)

        lrs = LearningRateScheduler(n_epochs)
        reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=10, verbose=1)
        early_stopping = EarlyStopping(monitor='loss', min_delta=0, patience=10, verbose=1)
        model.fit(X, Y, epochs=int(epochs), batch_size=self.batch_size, callbacks=[history, reduce_lr, early_stopping], verbose=1)

        self.write_history(model_name, history)

        model.save("./model_file/"+model_name)

if __name__ == "__main__":
    length = 13
    batch_size = 20
    input_w = 1
    #准备数据
    data_path = "./data/*.csv"
    X, Y = concat_data(data_path)
    input = Input((13, ))
    model = Models(batch_size, input)
    model.train(model.bp, X,Y, "bp.h5")

