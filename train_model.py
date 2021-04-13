from Data_process import concat_data
from model import Models
from keras.layers import Input
import sys

def judge(para1, model):
    if para1 == "lstm":
        train_model = model.lstm_model
    if para1 == "bp":
        train_model = model.bp
    if para1 == "rnn":
        train_model = model.rnn

    return train_model

if __name__ == "__main__":
    length = 13
    batch_size = 20
    input_w = 1
    # 准备数据
    data_path = "./data/*.csv"
    X, Y = concat_data(data_path)
    input = Input((13,))
    model = Models(batch_size, input)

    train_model = judge(sys.argv[1], model)

    model.train(train_model, X, Y, sys.argv[2], sys.argv[3])