import pandas as pd
import numpy as np
import json

train_mean, test_mean = 0, 0
train_std, test_std = 0, 0

pd_train = pd.read_csv("./AS1_data/train.csv","r")
pd_test = pd.read_csv("./AS1_data/test.csv","r")


train_x, train_y, test_x= [], [], []

for i in range(pd_train.shape[0]):
    tmp = []
    y, x = str(pd_train.iloc[i,0]).split(",")
    x = x.replace('"', '')
    for num in x.split(" "):
        tmp.append(int(num))
    train_x.append(tmp)
    train_y.append(int(y))
train_mean = np.mean(np.array(train_x))
train_std = np.std(np.array(train_std))
dict = {'emotion':train_y, 'pixels':train_x}
with open("train.json", "w") as f:
    f.write(json.dumps(dict, separators=(',',':')))


for i in range(pd_test.shape[0]):
    tmp = []
    x = str(pd_test.iloc[i,0])
    x = x.replace('"', '')
    x = x.replace(",", " ")
    for num in x.split(" "):
        tmp.append(int(num))
    test_x.append(tmp)
test_mean = np.mean(np.array(test_x))
test_std = np.std(np.array(test_x))
dict = {'pixels':test_x}
with open("test.json", "w") as f:
    f.write(json.dumps(dict, separators=(',',':')))
