# -*- coding: utf-8 -*-
#利用lstm预测
__author__ = 'fff_zrx'
import pandas as pd
import numpy as np
import random
import os
import keras
import pywt
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense,Dropout
from keras.layers import LSTM
from matplotlib import pyplot
from sklearn import preprocessing
from keras import backend as K
from keras.callbacks import ModelCheckpoint
import openpyxl
from keras import losses
from sklearn.decomposition import PCA
# ---- 数据导入 ----
data = pd.read_excel("./datas/traindata.xlsx")
origin_data_x = data.iloc[:,2:].values
origin_data_y=data.iloc[:,0].values
index = [j for j in range(len(origin_data_x))]
random.shuffle(index)
origin_data_y = origin_data_y[index]
origin_data_x = origin_data_x[index]
# ---- 参数定义----
split_point=int(len(origin_data_x)*0.8)
input_size=11 # 输入层维数
time_step =5 # 步长窗口
epochs=150
batch_size=72
# 标准化，工具函数
def calculate_mape(data_x,data_y):
    index = list(np.nonzero(data_y)[0])
    data_y = np.array([data_y[i] for i in index])
    predict= model.predict(data_x)
    predict = np.array([predict[i] for i in index])
    return np.mean(np.abs(data_y - predict) * std / (np.abs(data_y * std + mean)))
def calculate_mae(data_x, data_y):
    index = list(np.nonzero(data_y)[0])
    data_y = np.array([data_y[i] for i in index])
    predict = model.predict(data_x)
    predict = np.array([predict[i] for i in index])
    return np.mean(np.abs(data_y - predict) * std)
def mape(y_true, y_pred):
    return  K.mean(K.abs(y_true - y_pred)*std/(K.abs(y_true*std+mean)))
class mape_callback(keras.callbacks.Callback):
    def __init__(self, training_data, validation_data):
        self.x = training_data[0]
        self.y = training_data[1]
        self.x_val = validation_data[0]
        self.y_val = validation_data[1]
    def on_train_begin(self, logs={}):
        return
    def on_train_end(self, logs={}):
        return
    def on_epoch_begin(self, epoch, logs={}):
        return
    def on_epoch_end(self, epoch, logs={}):
        y_pred = self.model.predict(self.x)
        mape = np.mean(np.abs(self.y - y_pred)*std/ (np.abs(self.y*std+mean)))
        y_pred_val = self.model.predict(self.x_val)
        val_mape=np.mean(np.abs(self.y_val - y_pred_val)*std /(np.abs(self.y_val*std+mean)))
        print('mape: %s - val_mape: %s' % (mape,val_mape))
        return
    def on_batch_begin(self, batch, logs={}):
        return
    def on_batch_end(self, batch, logs={}):
        return

# 训练集数据
train_x= origin_data_x[:split_point]
scaler = preprocessing.StandardScaler().fit(train_x)
train_x=scaler.transform(train_x)
train_x=train_x.reshape([-1,input_size,time_step])
train_x=np.transpose(train_x,[0,2,1])
train_y = origin_data_y[:split_point]
train_y= train_y.reshape([-1,1])
scaler1 = preprocessing.StandardScaler().fit(train_y)
train_y=scaler1.transform(train_y)
mean=scaler1.mean_
std=np.sqrt(scaler1.var_)
# 测试集数据
test_x= origin_data_x[split_point:]
test_x=scaler.transform(test_x)
test_x = test_x.reshape([-1, input_size, time_step])
test_x = np.transpose(test_x, [0, 2, 1])
test_y = origin_data_y[split_point:]
test_y= test_y.reshape([-1,1])
test_y=scaler1.transform(test_y)
# design network
model = Sequential()
model.add(LSTM(30, input_shape=(train_x.shape[1], train_x.shape[2]),return_sequences=False))
# model.add(LSTM(30, return_sequences=False))
model.add(Dropout(0.5))
model.add(Dense(1))
model.summary() #打印出模型概况
model.compile(loss=["mae"], optimizer='adam',metrics=[mape])
# fit network
# filepath='model_trained.h5'
# # checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
history = model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size,  validation_data=[test_x, test_y],verbose=2, shuffle=True)
#save model after train保存模型文件
model.save('./models/lstm_model_number.h5')
# test the model
print("Testdatasets mape:",calculate_mape(test_x,test_y))
print("Testdatasets mae:",calculate_mae(test_x,test_y))
##plot history画出训练过程
pyplot.figure(1)
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.xlabel('Epochs', fontsize = 12)
pyplot.ylabel('Loss', fontsize = 12)
pyplot.savefig("Loss.png")
pyplot.show()
pyplot.figure(2)
pyplot.plot(history.history['mape'], label='train')
pyplot.plot(history.history['val_mape'], label='test')
pyplot.legend()
pyplot.xlabel('Epochs', fontsize = 12)
pyplot.ylabel('Mape', fontsize = 12)
pyplot.savefig("Mape.png")
pyplot.show()
# deletes the existing model
# del model
# load model
# model=load_model('./models/lstm_model_number.h5')
#---- 待预测数据导入 ----
data = pd.read_excel("./datas/testdata_for_number.xlsx")
x = data.iloc[:, 4:].values
x = scaler.transform(x)
x = x.reshape([-1, 11, 5])
x = np.transpose(x, [0, 2, 1])
y = data.iloc[:, 3].values
true = y.reshape([-1, 1])
t = data.iloc[:, 2]
# load model
model._make_predict_function()
predict = model.predict(x)
predict = scaler1.inverse_transform(predict)
# 计算mape与mae
index = list(np.nonzero(true)[0])
after_true = np.array([true[i] for i in index])
after_predict = np.array([predict[i] for i in index])
mape = np.mean(np.abs(after_true - after_predict) / (np.abs(after_true)))
mae = np.mean(np.abs(after_true - after_predict))
print("testday:3月12-15日  Mape:",mape)
print("testday:3月12-15日  Mae:",mae)
y=y.reshape([-1,])
analysis=[list(y),list(predict)]
mat=np.mat(analysis)
mat=mat.T
np.savetxt('./datas/test_data_number_output.csv',mat,delimiter=',')
print("写入预测数据成功！")
