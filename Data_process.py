import tqdm
import pandas as pd
import glob
import numpy as np
from sklearn.preprocessing import minmax_scale

def process_data(data_path, out_path):
    # 读入数据
    data = pd.read_csv(data_path, encoding='gb18030')

    # 构造充放电特征
    data['充电电流'] = 0
    data['放电电流'] = 0
    try:
        data.loc[data["电流/A"] > 0, "充电电流"] = abs(data.loc[data["电流/A"] > 0, "电流/A"])
        data.loc[data["电流/A"] < 0, "放电电流"] = abs(data.loc[data["电流/A"] < 0, "电流/A"])
    except:
        data.rename(columns={'电流': '电流/A'})
        data.loc[data["电流/A"] > 0, "充电电流"] = abs(data.loc[data["电流/A"] > 0, "电流/A"])
        data.loc[data["电流/A"] < 0, "放电电流"] = abs(data.loc[data["电流/A"] < 0, "电流/A"])

    # 构造充放电时间特征
    data['充电时间'] = 0
    data['放电时间'] = 0
    for i, x in enumerate(data["电流/A"]):
        if (i == 0) and (x < 0):
            data.loc[i, "放电时间"] = 1
        elif (i == 0) and (x > 0):
            data.loc[i, "充电时间"] = 0
        # 计算放电
        elif (i != 0) and (x < 0) and (data.loc[i - 1, "电流/A"] < 0):
            data.loc[i, "放电时间"] = data.loc[i - 1, "放电时间"] + 1
        elif (i != 0) and (x < 0) and (data.loc[i - 1, "电流/A"] > 0):
            data.loc[i, "放电时间"] = 1
        # 计算充电
        elif (i != 0) and (x > 0) and (data.loc[i - 1, "电流/A"] > 0):
            data.loc[i, "充电时间"] = data.loc[i - 1, "充电时间"] + 1
        elif (i != 0) and (x > 0) and (data.loc[i - 1, "电流/A"] < 0):
            data.loc[i, "充电时间"] = 1

    # 输出文件
    data.to_csv(out_path + data_path.split("/")[-1].split("\\")[-1].split(".")[0] + "f.csv")

def contribute_data(data_path):
    data = pd.read_csv(data_path, encoding="gb18030")
    X = data.loc[:, ["电流/A", "环境温度/度", "CH1/温度", "CH3/温度", "CH4/温度", "CH5/温度", "CH6/温度","CH7/温度",\
                     "CH8/温度", "充电电流", "放电电流", "充电时间", "放电时间"]]
    X = X.values[:-1]
    Y = data.loc[1:, ["CH1/温度", "CH3/温度", "CH4/温度", "CH5/温度", "CH6/温度","CH7/温度", "CH8/温度"]]
    return X, Y.values

def concat_data(data_path):
    csv_list = glob.glob(data_path)
    for i, csv in enumerate(tqdm.tqdm(csv_list)):
        X, Y = contribute_data(csv)
        if i == 0:
            X_train = X
            Y_label = Y
        else:
            X_train = np.concatenate((X_train, X), axis=0)
            Y_label = np.concatenate((Y_label, Y))

    X_train = minmax_scale(X_train)
    Y_label = minmax_scale(Y_label)

    #X_train = X_train.reshape(-1, 13, 1)
    # Y_label = Y_label.reshape(-1, 7)


    return X_train, Y_label

if __name__ == "__main__":
    data_path = "./data/*.csv"
    X, Y = concat_data(data_path)
    print(X.shape, Y.shape)