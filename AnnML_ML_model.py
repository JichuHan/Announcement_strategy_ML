#!/usr/bin/env python
# coding: utf-8

# In[2]:


import seaborn as sns
import multiprocessing as mp
import pandas as pd
import tqdm
import time
import copy
import numpy as np
#import MySQLdb
import os
import pickle
import warnings
import matplotlib.pyplot as plt

import pickle
def save_variable(v,filename):
    f=open(filename,'wb')          #打开或创建名叫filename的文档。
    pickle.dump(v,f)               #在文件filename中写入v
    f.close()                      #关闭文件，释放内存。
    return filename

def load_variable(filename):
    try:
        f=open(filename,'rb')
        r=pickle.load(f)
        f.close()
        return r
    except EOFError:
        return ""

pickle_path='/Users/jason/JC/公告数据/'
Factor_total_dict = load_variable(pickle_path+'Factors')


# 几个bug：
# 
# 1. 限制不允许卖空，以及其他条件 VS 按照最宽松的条件筛选，最后再进行调整。哪个更好？目前看后者更好，因为这样能最大限度利用信息，
# 
# 2. 当协方差矩阵非正定的时候，将协方差矩阵进行乘方直到能变成正定(奇数次方，这样方向不变顺序不变），是否是一种可行的方法
# 
# 3. 输入的参数很小的时候，可能在树模型里面会出现问题，这个时候扩大一定比例就可以减弱这个问题，这是为什么?
# 
# 对于Y，如果将它进行排序，可能会损失很多的信息，因此最好不排序。目前是直接归一化。
# 
# 对于X，因为很可能存在很多噪音，并且积少成多，因此直接排序，当然之后也可以尝试其他的方法，比如映射到正态分布上面等等
# 
# 但是树模型最好不要让X太小，我现在是*10，不然会存在分类完全一致这种会让IC失效的情况
# 
# XY的变化对模型本身有非常大的影响，需要谨慎选择。可以先利用最普通的模型进行训练，最后再对结果进行调整

# In[2]:


#save_variable(signal_df,'/Users/jason/JC/公告数据/signal_df')


# In[3]:


path = '/Users/jason/Desktop/公告类数据/'


# In[4]:


signal_df = load_variable(path+'signal_df')
twap_ret = load_variable(path + 'theoretical_rtn_metric')
twap_ret.index = [int(i) for i in twap_ret.index]
twap_ret.columns = [int(i) for i in twap_ret.columns]


# In[5]:


'''exer_df = pd.read_excel(path+'股权激励.xlsx')
twap_ret = pd.read_excel(path+'Twap_return.xlsx')
trade_vol = pd.read_excel(path+'交易量.xlsx')
ind_ret = pd.read_excel(path+'指数收益.xlsx')
buy_back_df = pd.read_excel(path+'回购.xlsx')
chg_df = pd.read_excel(path+'股东增持.xlsx')

twap_open = pd.read_excel(path+'TW开盘价.xlsx')
open_price = pd.read_excel(path+'开盘价.xlsx')
close_price = pd.read_excel(path+'收盘价.xlsx')
market_value = pd.read_excel(path+'市值.xlsx')
vwap = pd.read_excel(path+'VWAP.xlsx')

def treat_downloads(df):
    df = df.set_index('Unnamed: 0')
    df.columns = [int(i) for i in df.columns]
    return df
#twap_open = treat_downloads(twap_open)
#open_price = treat_downloads(open_price)
#close_price = treat_downloads(close_price)
#market_value = treat_downloads(market_value)
#vwap = treat_downloads(vwap)
#trade_vol = treat_downloads(trade_vol)
twap_ret = pd.read_excel(path+'Twap_return.xlsx')
twap_ret = treat_downloads(twap_ret)

save_variable(twap_ret,pickle_path + 'twap_ret')
twap_ret = read_variable(pickle_path + 'twap_ret')'''
1


# # 组合预测

# In[6]:


plt.plot(list(signal_df.count()))
plt.plot(list((twap_ret[signal_df.columns]*signal_df).count()))


# 构造Y，往前延展了60天，往后延展了一些时候

# In[7]:


real_ret = twap_ret[signal_df.columns]*signal_df


# In[8]:


'''date_input = list(signal_df.columns[66:-30])
for i in date_input:
    bo = real_ret[i].count() == len(model_main_single(Factor_total_dict,
                  real_ret,i,length=60,train_n_predict = train_n_predict_ols)[0])
    print(bo)'''
1


# In[93]:





# In[147]:


def sm_add_constant(testX):
    testX['const'] = np.ones(testX.shape[0])
    return testX


def train_n_predict_pca(trainX, trainY, testX, testY):
    Reg_df_adj=trainX.copy()
    Reg_df_adj['Y']=trainY.copy()
    Reg_df_adj_dropna=Reg_df_adj.dropna(axis=0)
    Y=Reg_df_adj_dropna['Y']
    X=Reg_df_adj_dropna.iloc[:,:-1]
    weights = np.linalg.eig(X.cov())[1]
    
    
    tX=testX
    tX=tX.fillna(0)
    predY = sum([np.dot(tX,weights[i]) for i in range(0,len(weights))])
    pred_test_Y = testY.copy(deep=True)
    pred_test_Y.iloc[:] = predY
    
    return pred_test_Y,weights

from sklearn import svm
def train_n_predict_svm(trainX, trainY, testX, testY):
    Reg_df_adj=trainX.copy()
    Reg_df_adj['Y']=trainY.copy()
    Reg_df_adj_dropna=Reg_df_adj.dropna(axis=0)
    Y=Reg_df_adj_dropna['Y']
    X=Reg_df_adj_dropna.iloc[:,:-1]
    clf=svm.SVR()  ##默认参数：kernel='rbf'
    clf.fit(X,Y)
    
    tX=testX
    tX=tX.fillna(0)
    predY = clf.predict(tX)
    pred_test_Y = testY.copy(deep=True)
    pred_test_Y.iloc[:] = predY
    
    return pred_test_Y,clf

def train_n_predict_en(trainX, trainY, testX, testY):
    Reg_df_adj=trainX.copy()
    Reg_df_adj['Y']=trainY.copy()
    Reg_df_adj_dropna=Reg_df_adj.dropna(axis=0)
    Y=Reg_df_adj_dropna['Y']
    X=pd.DataFrame(sm_add_constant(Reg_df_adj_dropna.iloc[:,:-1]))
    model = ElasticNet(random_state=0,alpha=0.001,l1_ratio=0.5)
    model.fit(X, Y) 
    
    tX=pd.DataFrame(sm_add_constant(testX))
    tX=tX.fillna(0)
    predY = model.predict(tX)
    pred_test_Y = testY.copy(deep=True)
    pred_test_Y.iloc[:] = predY
        
    return pred_test_Y,model

import statsmodels.api as sm
from sklearn.linear_model import ElasticNet
def train_n_predict_en(trainX, trainY, testX, testY):
    Reg_df_adj=trainX.copy()
    Reg_df_adj['Y']=trainY.copy()
    Reg_df_adj_dropna=Reg_df_adj.dropna(axis=0)
    Y=Reg_df_adj_dropna['Y']
    X=pd.DataFrame(sm_add_constant(Reg_df_adj_dropna.iloc[:,:-1]))
    model = ElasticNet(random_state=0,alpha=0.001,l1_ratio=0.5)
    model.fit(X, Y) 
    
    tX=pd.DataFrame(sm_add_constant(testX))
    tX=tX.fillna(0)
    predY = model.predict(tX)
    pred_test_Y = testY.copy(deep=True)
    pred_test_Y.iloc[:] = predY
        
    return pred_test_Y,model


import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import *
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping

def train_n_predict_lstm(trainX, trainY, testX, testY):
    Reg_df_adj=trainX.copy()
    Reg_df_adj['Y']=trainY.copy()
    Reg_df_adj_dropna=Reg_df_adj.dropna(subset = ['Y']).fillna(0)
    sc = MinMaxScaler(feature_range = (0, 1))
    Reg_df_scaled = sc.fit_transform(Reg_df_adj_dropna)
    
    Y=Reg_df_scaled[:,-1]
    X=Reg_df_scaled[:,:-1]
    X_self = np.reshape(X, (X.shape[0], X.shape[1], 1))
    Y_self = Y
    
    model = Sequential()
    #Adding the first LSTM layer and some Dropout regularisation
    model.add(LSTM(units = 20, return_sequences = True, input_shape = (X_self.shape[1], 1)))
    model.add(Dropout(0.2))
    # Adding a second LSTM layer and some Dropout regularisation
    model.add(LSTM(units = 20, return_sequences = True))
    model.add(Dropout(0.2))
    # Adding a third LSTM layer and some Dropout regularisation
    model.add(LSTM(units = 20, return_sequences = True))
    model.add(Dropout(0.2))
    # Adding a fourth LSTM layer and some Dropout regularisation
    model.add(LSTM(units = 20))
    model.add(Dropout(0.2))
    # Adding the output layer
    model.add(Dense(units = 1))
    # Compiling the RNN
    model.compile(optimizer = 'adam', loss = 'mean_squared_error')
    # Fitting the RNN to the Training set
    model.fit(X_self, Y_self, epochs = 10, batch_size = 64)
    
    testX_scaled = sc.fit_transform(testX)
    X_test = np.reshape(np.array(testX_scaled), (np.array(testX_scaled).shape[0], np.array(testX_scaled).shape[1], 1))
    predicted_Y_scaled = model.predict(X_test)
    predicted_Y = pd.DataFrame(predicted_Y_scaled)
    pred_test_Y = testY.copy(deep=True)
    pred_test_Y.iloc[:] = predicted_Y[0]
        
    return pred_test_Y.fillna(predicted_Y.mean()),np.nan


reg_xgb_paras_boost = {#'num_round': 5, #不能写num_round这个参数？？？？？？？？？？？？？？？？？？？
                    #'num_boost_round':5,
                    'max_depth': 5,
                    'eta': 0.2,
                    'subsample': 1,  # 随机采样训练样本 训练实例的子采样比
                    #'gamma': 0.15,  # 用于控制是否后剪枝的参数,越大越保守，一般0.1-0.2
                    'reg_lambda': 0.7,  # 0.1控制模型复杂度的权重值的L2正则化项参数
                    'reg_alpha':0.001,
                    'max_delta_step': 0,
                    'colsample_bytree': 0.8,  # 生成树时进行的列采样
                    'booster': 'gbtree',
                    #'tree_method': 'gpu_hist',
                    'objective': 'reg:squarederror',
'subsample':0.8}

reg_xgb_paras ={
        #'num_boost_round': 100,
    #reg_xgb_paras_boost['num_round'],
        'verbose_eval': False,
        #'maximize': False,
        'params': reg_xgb_paras_boost}

params={'booster':'gbtree',
        'nthread':12,
        'objective': 'rank:pairwise',
        'eval_metric':'auc',
        'seed':0,
        'eta': 0.01,
        'gamma':0.1,
        'min_child_weight':1.1,
        'max_depth':5,
        'lambda':10,
        'subsample':0.7,
        'colsample_bytree':0.7,
        'colsample_bylevel':0.7,
        'tree_method':'exact'
        }

params_empty={}

def train_n_predict_xgb3(trainX, trainY, testX, testY):
    import xgboost as xgb
    
    train_xgb = xgb.DMatrix(data=trainX, label=trainY.replace([-np.inf,np.inf],[0,0]).fillna(0))
    test_xgb = xgb.DMatrix(data=testX, label=testY.replace([-np.inf,np.inf],[0,0]).fillna(0))
    
    #XgbTree = xgb.train(**reg_xgb_paras,dtrain=train_xgb,num_boost_round=1000)
    #XgbTree = xgb.train(params,train_xgb,num_boost_round=1000)
    XgbTree = xgb.train(params_empty,train_xgb)
    
    pred_test_y = np.array(XgbTree.predict(test_xgb, ntree_limit=XgbTree.best_ntree_limit))
    
    pred_test_Y = testY.copy(deep=True)
    pred_test_Y.iloc[:] = pred_test_y
    
    return pred_test_Y, XgbTree

def train_n_predict_xgb1(trainX, trainY, testX, testY):
    import xgboost as xgb
    
    train_xgb = xgb.DMatrix(data=trainX, label=trainY.replace([-np.inf,np.inf],[0,0]).fillna(0))
    test_xgb = xgb.DMatrix(data=testX, label=testY.replace([-np.inf,np.inf],[0,0]).fillna(0))
    
    XgbTree = xgb.train(**reg_xgb_paras,dtrain=train_xgb,num_boost_round=1000)
    #XgbTree = xgb.train(params,train_xgb,num_boost_round=1000)
    #XgbTree = xgb.train(params_empty,train_xgb)
    
    pred_test_y = np.array(XgbTree.predict(test_xgb, ntree_limit=XgbTree.best_ntree_limit))
    
    pred_test_Y = testY.copy(deep=True)
    pred_test_Y.iloc[:] = pred_test_y
    
    return pred_test_Y, XgbTree

def train_n_predict_xgb2(trainX, trainY, testX, testY):
    import xgboost as xgb
    
    train_xgb = xgb.DMatrix(data=trainX, label=trainY.replace([-np.inf,np.inf],[0,0]).fillna(0))
    test_xgb = xgb.DMatrix(data=testX, label=testY.replace([-np.inf,np.inf],[0,0]).fillna(0))
    
    #XgbTree = xgb.train(**reg_xgb_paras,dtrain=train_xgb,num_boost_round=1000)
    XgbTree = xgb.train(params,train_xgb,num_boost_round=1000)
    #XgbTree = xgb.train(params_empty,train_xgb)
    
    pred_test_y = np.array(XgbTree.predict(test_xgb, ntree_limit=XgbTree.best_ntree_limit))
    
    pred_test_Y = testY.copy(deep=True)
    pred_test_Y.iloc[:] = pred_test_y
    
    return pred_test_Y, XgbTree

import statsmodels.api as sm
def train_n_predict_ols(trainX, trainY, testX, testY):
    Reg_df_adj=trainX.copy()
    Reg_df_adj['Y']=trainY.copy()
    Reg_df_adj_dropna=Reg_df_adj.dropna(axis=0)
    Y=Reg_df_adj_dropna['Y']
    X=pd.DataFrame(sm_add_constant(Reg_df_adj_dropna.iloc[:,:-1]))
    model = sm.OLS(Y, X).fit() 
    
    tX=pd.DataFrame(sm_add_constant(testX))
    tX=tX.fillna(0)
    predY = model.predict(tX)
    pred_test_Y = testY.copy(deep=True)
    pred_test_Y.iloc[:] = predY
        
    return pred_test_Y,model

def train_n_predict_ew(trainX, trainY, testX, testY):
    pred_test = testX.mean(axis = 1)
    pred_test_Y = testY.copy(deep=True)
    pred_test_Y.iloc[:] = pred_test
    model = np.nan
    return pred_test_Y,model

def train_n_predict_ols(trainX, trainY, testX, testY):
    #print(testY)
    Reg_df_adj=trainX.copy()
    Reg_df_adj['Y']=trainY.copy()
    Reg_df_adj_dropna=Reg_df_adj.dropna(axis=0)
    Y=Reg_df_adj_dropna['Y']
    X=pd.DataFrame(sm_add_constant(Reg_df_adj_dropna.iloc[:,:-1]))
    model = sm.OLS(Y, X).fit() 
    
    tX=pd.DataFrame(sm_add_constant(testX))
    tX=tX.fillna(0)
    predY = model.predict(tX)
    pred_test_Y = testY.copy(deep=True)
    #print(pred_test_Y)
    pred_test_Y.iloc[:] = predY
        
    return pred_test_Y,model


def train_n_predict_his(trainX, trainY, testX, testY):
    trainY_mean = trainY.unstack().mean()
    
    testY_df = testY.unstack()
    a=[list(trainY_mean)]
    test_pred_df = pd.DataFrame(a*len(testY_df),columns = testY_df.columns,index = testY_df.index)
    
    predY = test_pred_df.stack()
    pred_test_Y = testY.copy(deep=True)
    pred_test_Y.iloc[:] = predY
     
    model = np.nan
    return pred_test_Y,model


from sklearn.ensemble import RandomForestRegressor
def train_n_predict_rf(trainX, trainY, testX, testY):
    estimator=RandomForestRegressor(n_estimators=100,
                                    max_depth=5,min_samples_split=5,max_leaf_nodes=20,
                                    oob_score=True,random_state=1)
    Reg_df_adj=trainX.copy()
    Reg_df_adj['Y']=trainY.copy()
    Reg_df_adj_dropna=Reg_df_adj.dropna(axis=0)
    Y=Reg_df_adj_dropna['Y']
    X=Reg_df_adj_dropna.iloc[:,:-1]
    
    estimator.fit(X,Y)
    
    test_adj=testX.fillna(0)
    predY = estimator.predict(test_adj)
    
    pred_test_Y = testY.copy(deep=True)
    pred_test_Y.iloc[:] = predY
 
    return pred_test_Y,estimator

from sklearn.neighbors import KNeighborsRegressor
def train_n_predict_knn(trainX, trainY, testX, testY):
    Reg_df_adj=trainX.copy()
    Reg_df_adj['Y']=trainY.copy()
    Reg_df_adj_dropna=Reg_df_adj.dropna(axis=0)
    Y=Reg_df_adj_dropna['Y']
    X=Reg_df_adj_dropna.iloc[:,:-1]
    #print(X.shape)
    knn = KNeighborsRegressor()
    knn.fit(X,Y)
    test_adj=testX.fillna(0)
    #print(test_adj.shape)
    predY = knn.predict(test_adj)
    pred_test_Y = testY.copy(deep=True)
    pred_test_Y.iloc[:] = predY
   
    return pred_test_Y,knn


# 训练用模型

# In[148]:


def get_train_df(date,factor,length=60):
    daily_index = list(real_ret[date].dropna().index)
    posi_temp = list(factor.columns).index(date)
    return factor.loc[daily_index].iloc[:,posi_temp-length:posi_temp+1]

def get_factor_data(factor_data_dict, Return,factor_method='raw'):

    def transform_to_2d(pqidata):
        shape = (pqidata.shape[1]*pqidata.shape[2], pqidata.shape[0])
        return pqidata.transpose((2, 1, 0)).reshape(shape)

    factor_dict = {}

    for fac_name in factor_data_dict.keys():
        if factor_method == 'raw':
            factor_dict[fac_name] = factor_data_dict[fac_name]
        elif factor_method == 'linear':
            factor_dict[fac_name] = factor_data_dict[fac_name].rank(pct=True) - 0.5
        else:
            factor_dict[fac_name] = factor_data_dict[fac_name]

    idx = list(factor_data_dict.values())[0].unstack().index
    factor_data = pd.DataFrame(transform_to_2d(pqidata=np.array(list(factor_dict.values()))))
    factor_data.columns = factor_data_dict.keys()
    factor_data.index = idx
    #.rank(pct=True)-0.5
    Return_stacked = ((Return-Return.min())/(Return.max()-Return.min())-0.5).T.stack(dropna=False)

    return factor_data,Return_stacked

def get_Xdata_Y(factor_sta, return_sta, date_list):
    YData = return_sta.loc[date_list]
    XData = factor_sta.loc[date_list]
    return XData, YData


#输入rawdata，就是一个dict一个return_df，然后是date（需要预测的日期），以及训练的长度（目前是60天），训练函数
def model_main_single(factor_dict,return_df,date=20180102,length=60,train_n_predict=train_n_predict_xgb1):
    
    factor_part_dict = {i:get_train_df(date,factor_dict[i],length) for i in factor_dict}
    return_part_df = get_train_df(date,return_df,length)
    
    factor_part_sta,return_part_sta = get_factor_data(factor_part_dict, return_part_df,factor_method='linear')
    train_test_date = list(return_part_df.columns)
    #split_loc = int(len(train_test_date)*0.7)
    train_date = train_test_date[:-1]
    test_date = train_test_date[-1]
    #print(test_date)
    #print(train_date)
    #print(test_date)
    trainX, trainY = get_Xdata_Y(factor_part_sta, return_part_sta, train_date)
    testX, testY = get_Xdata_Y(factor_part_sta, return_part_sta, test_date)
    pred_test_Y,model = train_n_predict(trainX, trainY, testX, testY)

    #print(return_part_df.columns)
    return pred_test_Y, model

def main_loop(date_list,factor_dict,return_df,length,train_n_predict = 1):
    prediction_dict = {}
    model_dict = {}
    for i in tqdm.tqdm(range(0,len(date_list))):
        print(date_list[i])
        pred_test_Y, model = model_main_single(factor_dict,return_df,date_list[i],length,train_n_predict)
        print(pred_test_Y)
        prediction_dict[date_list[i]] = pred_test_Y
        model_dict[date_list[i]] = model
    return prediction_dict,model_dict

date_input = real_ret.columns[60:-20]
def train_model(train_n_predict_ols):
    pred_ols,mod_ols = main_loop(date_input,Factor_total_dict,
                      real_ret,length=60,train_n_predict = train_n_predict_ols)
    '''ic_lst = []
    for i in range(0,len(date_input)):
        ic = pd.DataFrame([pred_ols[date_input[i]],real_ret[date_input[i]].dropna()]).T.corr().iloc[1,0]
        ic_lst.append(ic)
    plt.plot([len(pred_ols[i]) for i in pred_ols])
    plt.show()
    plt.plot(ic_lst)
    plt.show()
    print(np.mean(ic_lst))'''
    return pred_ols,mod_ols
    


# In[ ]:


real_ret.columns[60:-20]


# 循环进行预测

# 组合权重优化

# In[149]:


from pypfopt import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns
from pypfopt import plotting

def port_opt(Return,Return_forcast_dict,is_co = False):
    #cov_mat = twap_ret.iloc[:,:1460].loc[Factor_total_dict[list(Factor_total_dict.keys())[0]].index].T.cov()
    #cov_mat = Return.T.cov()
    #variance = cov_mat
    #weight_bounds可以调整是否允许卖空
    weights_dict = {}
    date_list = list(Return_forcast_dict.keys())
    for i in tqdm.tqdm(range(0,len(Return_forcast_dict.keys()))):
        stock_list = list(Return_forcast_dict[date_list[i]].index)
        mean = Return_forcast_dict[date_list[i]]
        #if is_co:
        #    ef = EfficientFrontier(mean, variance,weight_bounds=(0, 0.5))
        #    print(1)
        #else:
        date_index = list(Return.columns).index(date_list[i])
        partial_ret = Return.loc[stock_list].iloc[:,(date_index-60):date_index]
        partial_ret = partial_ret.T.fillna(partial_ret.mean(axis=1)).T.fillna(0)
        variance = partial_ret.T.cov()
        
        for j in range(1,10,2):
            try:
                ef = EfficientFrontier(mean,variance**j,weight_bounds=(-1, 1))
                weight = ef.max_sharpe(risk_free_rate=0.001)
                weight_df = pd.DataFrame(weight,index=['Weight'])
                print(j)
                break
            except:
                if j == 9:
                    print('Error '+str(date_list[i]))
                    weight_df = pd.DataFrame(np.ones(len(stock_list))*(1/len(stock_list)),index = stock_list,columns = ['Weight']).T
                continue
            '''try:
                ef = EfficientFrontier(mean,variance**2,weight_bounds=(-1, 1))
                weight = ef.max_sharpe(risk_free_rate=0.001)
                weight_df = pd.DataFrame(weight,index=['Weight'])
            except:
                try:
                    ef = EfficientFrontier(mean,variance**3,weight_bounds=(-1, 1))
                    weight = ef.max_sharpe(risk_free_rate=0.001)
                    weight_df = pd.DataFrame(weight,index=['Weight'])
                except:
                    print('Error '+str(date_list[i]))
                    weight_df = pd.DataFrame(np.ones(len(stock_list))*(1/len(stock_list)),index = stock_list,columns = ['Weight']).T'''
        #print(list(Return_forcast_dict.keys())[i])
        weights_dict[list(Return_forcast_dict.keys())[i]] = weight_df
    
    return weights_dict
    '''weights_df = pd.concat(weights)
    weights_df.rolling(window=20).mean().plot(figsize=(14,4))
    plt.show()

    Return_port = (weights_df * Return.T).sum(axis=1)
    Return_port.cumsum().plot(figsize=(14,4))
    plt.show()
    
    return weights_df'''


# 权重优化循环

# In[150]:


def portfolio_calculator(pred_ols):
    wd = port_opt(real_ret,pred_ols,is_co = False)
    Ret_all_ew = []
    for i in wd.keys():
        Ret_all_ew.append((real_ret.loc[wd[i].columns][i]).mean())
    for i in wd:
        t = wd[i]
        t[t<=0] = 0
        wd[i] = t/float(np.sum(t,axis=1))
    Ret_all = []
    for i in wd.keys():
        Ret_all.append(((real_ret.loc[wd[i].columns][i]) * (wd[i].loc['Weight'])).sum())
    plt.figure(figsize=(15,8))
    plt.plot(list(pd.DataFrame(Ret_all).cumsum()[0]),color = 'blue')
    plt.plot(list(pd.DataFrame(Ret_all_ew).cumsum()[0]),color = 'red')
    plt.legend(['OLS+MV_OPT','Equal Weight'])
    plt.show()
    
    return wd,Ret_all


# In[151]:


pred_lstm,mod_lstm = train_model(train_n_predict_lstm)


# In[156]:


pred_lstm = {i:pred_lstm[i].fillna(pred_lstm[i].mean()) for i in pred_lstm}


# In[157]:


weight_lstm ,ret_lstm = portfolio_calculator(pred_lstm)


# In[ ]:





# In[111]:


mod_svm_s = mod_svm.copy()


# In[ ]:


pred_svm,mod_svm = train_model(train_n_predict_svm)


# In[ ]:


weight_svm ,ret_svm = portfolio_calculator(pred_svm)


# In[ ]:


pred_pca,mod_pca = train_model(train_n_predict_pca)


# In[ ]:


weight_pca ,ret_pca = portfolio_calculator(pred_pca)


# In[ ]:


pred_en,mod_en = train_model(train_n_predict_en)


# In[ ]:


weight_en ,ret_en = portfolio_calculator(pred_en)


# In[ ]:


pred_knn,mod_knn = train_model(train_n_predict_knn)


# In[ ]:


weight_knn ,ret_knn = portfolio_calculator(pred_knn)


# In[ ]:


pred_ols,mod_ols = train_model(train_n_predict_ols)


# In[ ]:


weight_ols ,ret_ols = portfolio_calculator(pred_ols)


# In[ ]:


pred_ew,mod_ew = train_model(train_n_predict_ew)


# In[ ]:


weight_ew ,ret_ew = portfolio_calculator(pred_ew)


# In[53]:


#pred_xgb1,mod_xgb1 = train_model(train_n_predict_xgb1)


# In[ ]:


pred_xgb2,mod_xgb2 = train_model(train_n_predict_xgb2)


# In[ ]:


weight_xgb2 ,ret_xgb2 = portfolio_calculator(pred_xgb2)


# In[56]:


#pred_xgb3,mod_xgb3 = train_model(train_n_predict_xgb3)


# In[ ]:


pred_rf,mod_rf = train_model(train_n_predict_rf)


# In[ ]:


weight_rf ,ret_rf = portfolio_calculator(pred_rf)


# In[ ]:





# In[ ]:





# In[68]:


v = {'pred':pred_ew,'model':mod_ew,'weight':weight_ew,'return':ret_ew}
save_variable(v,pickle_path + 'EW_dict')


# In[78]:


v = {'pred':pred_rf,'model':mod_rf,'weight':weight_rf,'return':ret_rf}
v['model'][20180102]
def FI_rf(v):
    return pd.DataFrame(v.feature_importances_,index = Factor_total_dict.keys(),columns = ['FI'])
v['model'] = {i:FI_rf(v['model'][i]) for i in v['model'].keys()}
save_variable(v,pickle_path + 'RF_dict')


# In[88]:


v = {'pred':pred_xgb2,'model':mod_xgb2,'weight':weight_xgb2,'return':ret_xgb2}
def FI_xgb(v):
    return pd.DataFrame(v.get_fscore(),index=['FI']).T
v['model'] = {i:FI_xgb(v['model'][i]) for i in v['model'].keys()}
save_variable(v,pickle_path + 'XGB_dict')


# In[96]:


v = {'pred':pred_en,'model':mod_en,'weight':weight_en,'return':ret_en}
def FI_en(v):
    return pd.DataFrame(v.coef_[:-1],index = Factor_total_dict.keys(), columns = ['FI'])
v['model'] = {i:FI_en(v['model'][i]) for i in v['model'].keys()}
save_variable(v,pickle_path + 'EN_dict')


# In[98]:


v = {'pred':pred_knn,'model':mod_knn,'weight':weight_knn,'return':ret_knn}
v['model'] = {i:np.nan for i in v['model'].keys()}
save_variable(v,pickle_path + 'KNN_dict')


# In[99]:


v = {'pred':pred_ols,'model':mod_ols,'weight':weight_ols,'return':ret_ols}
def FI_ols(v):
    return pd.DataFrame(v.params.iloc[:-1],columns = ['FI'])
v['model'] = {i:FI_ols(v['model'][i]) for i in v['model'].keys()}
save_variable(v,pickle_path + 'OLS_dict')


# In[ ]:





# In[125]:


v = {'pred':pred_svm,'model':mod_svm_s,'weight':weight_svm,'return':ret_svm}
def FI_svm(v):
    return pd.DataFrame(v.coef_[0],index = Factor_total_dict.keys(),columns = ['FI'])
v['model'] = {i:FI_svm(v['model'][i]) for i in v['model'].keys()}
save_variable(v,pickle_path + 'SVM_dict')


# In[131]:


v = {'pred':pred_pca,'model':mod_pca,'weight':weight_pca,'return':ret_pca}
def FI_pca(v):
    return pd.DataFrame(v[0],index = Factor_total_dict.keys(),columns = ['FI'])
v['model'] = {i:FI_pca(v['model'][i]) for i in v['model'].keys()}
save_variable(v,pickle_path + 'PCA_dict')


# In[158]:


v = {'pred':pred_lstm,'model':mod_lstm,'weight':weight_lstm,'return':ret_lstm}
v['model'] = {i:np.nan for i in v['model'].keys()}
save_variable(v,pickle_path + 'LSTM_dict')


# In[332]:


ic_lst_ols = []
for i in range(0,len(date_input)):
    ic = pd.DataFrame([pred_ols[date_input[i]],real_ret[date_input[i]].dropna()]).T.corr().iloc[1,0]
    ic_lst_ols.append(ic)
    
ic_lst_xgb = []
for i in range(0,len(date_input)):
    ic = pd.DataFrame([pred_xgb[date_input[i]],real_ret[date_input[i]].dropna()]).T.corr().iloc[1,0]
    ic_lst_xgb.append(ic)
    
ic_lst_rf = []
for i in range(0,len(date_input)):
    ic = pd.DataFrame([pred_rf[date_input[i]],real_ret[date_input[i]].dropna()]).T.corr().iloc[1,0]
    ic_lst_rf.append(ic)


# In[333]:


plt.figure(figsize=(15,5))
plt.plot(pd.DataFrame(ic_lst_ols).rolling(window=20).mean()[0])
plt.plot(pd.DataFrame(ic_lst_xgb).rolling(window=20).mean()[0])
plt.plot(pd.DataFrame(ic_lst_rf).rolling(window=20).mean()[0])
plt.legend(['OLS','XGB','RF'])


# In[230]:




