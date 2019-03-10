# -*- coding: utf-8 -*-
"""
Created on Mon Jul  2 23:18:45 2018

@author: lcy
"""

# -*- coding: utf-8 -*-
#运行时间的检测
import datetime
starttime = datetime.datetime.now()


import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import auc 
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

# 因为自变量数据有定类数据，所以把自变量进行标签化处理
def Encode(data):
    enc = preprocessing.LabelEncoder()
    columns = ['职业类型','省份', '教育', '家庭角色', '婚姻状况', '民族', '工作情况', '性别']#这些变量都属于定类变量，要进行标签化处理
    for i in columns:
      enc.fit(data[i])
      data[i] = enc.transform(data[i])

# 训练
def Training(train):
    Encode(train)  #对数据的类别字段进行编码
    train_xy,offline_test = train_test_split(train,test_size=0.2)  #将数据分为测试集和训练集
    train,val = train_test_split(train_xy, test_size = 0.2)  #将训练集分为训练集和验证集
    train_y = train.Y  #取出训练的因变量结果：Y
    train_X = train.drop(['Y'],axis=1)  #将除了Y外的数据都作为自变量
    val_y = val.Y
    val_X = val.drop(['Y'],axis=1)
    test_y = offline_test.Y
    test_x = offline_test.drop(['Y'],axis=1)
    lgb_train = lgb.Dataset(train_X,train_y,free_raw_data=False)   
    lgb_eval = lgb.Dataset(val_X,val_y,free_raw_data=False)
    params = {                            #设置lightgbm的参数
                'boosting_type': 'gbdt',  #使用gbdt来训练含有定类变量的二分模型
                  'boosting': 'dart',     #boosting的参数有许多，dart相对于其他的可以略微提高精确度
                  'objective': 'binary',  #要求输出的Y是一个二分类变量，因此objective参数设置为binary二分类
                  'metric': 'auc',        #评价标准：sklearn中的auc函数
      
                  'learning_rate': 0.01,  #学习率一般设置在0.01~0.001之间
                  'num_leaves':64,        #取值应 <= 2 ^（max_depth）， 超过此值可能会导致过拟合
                  'max_depth':6,          #树的最大深度，一般设置为6
      
                  'max_bin':100,          #如果要求精确度，设置大一些，要求速度，设置小一些
                  'min_data_in_leaf':200, #将它设置为较大的值可以避免生长太深的树，但可能会导致 underfitting，在大型数据集时就设置为数百或数千
      
                  'feature_fraction': 0.6,#防止过拟合
                  'bagging_fraction': 1,
                  'bagging_freq':0,
      
                  'lambda_l1': 0,         #默认值 0
                  'lambda_l2': 0,         #默认值 0
                  'min_split_gain': 0     #默认值 0
    }
    model = lgb.train(params,lgb_train,num_boost_round=1000,valid_sets=lgb_eval,early_stopping_rounds=30)  
    #开始训练！！！迭代次数1000次，如果有三十次没有提高就停止
    y_pred = model.predict(test_x)   
    fpr, tpr, thresholds = metrics.roc_curve(list(test_y), y_pred)
    AUC = auc(fpr, tpr)
    print(AUC)
    return model

# 预测
def Predicting(test,model):
    Encode(test)#预测文件test.csv
    y_pred = gbm.predict(test)#运用之前训练好的模型进行预测
    predictY = pd.DataFrame(y_pred.reshape(10000,1))#将预测好的10000条Y变成一个10000行×1列的数据框
    predictY.to_csv('Results_1.csv', encoding = 'utf-8', index=False , header=False) #结果以csv的形式输出

if __name__ == "__main__":
    train = pd.read_csv('Train.csv')
    test = pd.read_csv('Test.csv') 
    gbm = Training(train)
    Predicting(test,gbm)

    
#运行时间的检测#    
endtime = datetime.datetime.now()
print ("the total time is ",endtime - starttime,'seconds')