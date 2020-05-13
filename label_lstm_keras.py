# -*- coding: utf-8 -*-
#利用lstm分类
__author__ = 'fff_zrx'
import pandas as pd
import numpy as np
from numpy import array
import random
from sklearn.preprocessing import OneHotEncoder
from sklearn import preprocessing
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense, Dropout
from keras.layers import LSTM
from keras import regularizers
from matplotlib import pyplot
import openpyxl
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# ---- 数据导入 ----
data = pd.read_excel("./datas/traindata.xlsx")
origin_data_x = data.iloc[:,2:].values
origin_data_y=data.iloc[:,1].values
index = [j for j in range(len(origin_data_x))]
random.shuffle(index)
origin_data_y = origin_data_y[index]
origin_data_x = origin_data_x[index]
# ---- 参数定义----
split_point=int(len(origin_data_x)*0.8)
input_size=11
time_step =5
labels=5
epochs=200
batch_size=72
# 标准化，工具函数
def normal(data):
    mean=np.mean(data,axis=0)
    std=np.std(data,axis=0)
    return (data-mean)/std
#对labels进行one-hot编码
def label2hot(labels):
    values = array(labels)
    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = values.reshape(len(values), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
    return onehot_encoded
hot_data_y=label2hot(origin_data_y[:])
#hot_data_y.append(onehot_encoded)
#hot_data_y=array(hot_data_y).transpose((1,0,2))
# 训练集数据
train_x= origin_data_x[:split_point]
scaler = preprocessing.StandardScaler().fit(train_x)
train_x=scaler.transform(train_x)
# train_x = normal(train_x)
train_x=train_x.reshape([-1,input_size,time_step])
train_x=np.transpose(train_x,[0,2,1])
train_y = hot_data_y[:split_point]
# 测试集数据
test_x= origin_data_x[split_point:]
test_x=scaler.transform(test_x)
# test_x = normal(test_x)
test_x=test_x.reshape([-1,input_size,time_step])
test_x=np.transpose(test_x,[0,2,1])
test_y = hot_data_y[split_point:]
print("Data processing is finished!")
# design network
model = Sequential()
# model.add(LSTM(30, input_shape=(train_x.shape[1], train_x.shape[2]),kernel_regularizer=regularizers.l2(0.001),activity_regularizer=regularizers.l1(0.001)))
model.add(LSTM(30, input_shape=(train_x.shape[1], train_x.shape[2])))
# model.add(Dropout(0.2))
# model.add(LSTM(6, return_sequences=False))
model.add(Dense(labels, activation='softmax'))
model.summary() #打印出模型概况
model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
# fit network
history = model.fit(train_x,train_y, epochs=epochs, batch_size=batch_size, validation_data=(test_x, test_y), verbose=2, shuffle=True)
#save model after train保存模型文件
model.save('./models/lstm_model_label.h5')
# test the model
score = model.evaluate(test_x, test_y, verbose=2) #evaluate函数按batch计算在某些输入数据上模型的误差
print('Test accuracy:', score[1])
score = model.evaluate(train_x, train_y, verbose=2) #evaluate函数按batch计算在某些输入数据上模型的误差
print('Train accuracy:', score[1])
#导出数据
prediction_label = model.predict_classes(test_x)
prediction_label=[i+1 for i in prediction_label]
fact_label=np.argmax(test_y,1)
fact_label=[i+1 for i in fact_label]
analysis=[fact_label, prediction_label]
wb = openpyxl.Workbook()
sheet = wb.active
sheet.title = 'analysis_data'
for i in range(0, 2):
    for j in range(0, len(analysis[i])):
        sheet.cell(row=j + 1, column=i + 1, value=analysis[i][j])
wb.save('./datas/analysis_label.xlsx')
print("写入预测数据成功！")
# plot history
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.xlabel('Epochs', fontsize = 12)
pyplot.ylabel('Loss', fontsize = 12)
pyplot.savefig("./images/Loss_label.png")
pyplot.show()
pyplot.plot(history.history['acc'], label='train')
pyplot.plot(history.history['val_acc'], label='test')
pyplot.legend()
pyplot.xlabel('Epochs', fontsize = 12)
pyplot.ylabel('Accuracy', fontsize = 12)
pyplot.savefig("./images/Accuracy_label.png")
pyplot.show()
# deletes the existing model
#del model
# load model从模型文件加载模型
#model=load_model('./models/lstm_model_label.h5')
