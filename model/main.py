import numpy as np
import pandas as pd
from sklearn_pandas import DataFrameMapper
from pycox.preprocessing.feature_transforms import OrderedCategoricalLong
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from scipy import ndimage
from sklearn import metrics
from lifelines.utils import concordance
from pycox.evaluation import EvalSurv
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.optimizers import *
from tensorflow.keras import backend as k
from tensorflow.keras.regularizers import L1L2
from tensorflow.keras.callbacks import *
from pycox.preprocessing.label_transforms import LabTransDiscreteTime
import os
import sys
import tensorflow as tf
from tensorflow.python.ops import tensor_array_ops
np.random.seed(1234)
tf.random.set_seed(2020)
def logloss(lambda3):
    def loss(y_true, y_pred):
        mask_dead = y_true[:, 1]
        mask_alive = y_true[:, 0]
        mask_censored = 1 - (mask_alive + mask_dead)
        logloss = -1 * k.mean(mask_dead * k.log(y_pred[:, 1]) + mask_alive * k.log(y_pred[:, 0]))
        - lambda3 * k.mean(y_pred[:, 1] * mask_censored * k.log(y_pred[:, 1]))
        return logloss

    return loss

def rankingloss(y_true, y_pred, name=None):
    ranking_loss = 0
    for i in range(time_length):
        for j in range(i + 1, time_length, 1):
            tmp1  = y_pred[:, j] - y_pred[:, i]
            tmp2 = y_true[:, j] - y_true[:, i]
            tmp=tf.cast(tmp2,tf.float32)-tf.cast(tmp1,tf.float32)
            tmp3=tmp>0
            tmp3=tf.cast(tmp3, tf.float32)
            ranking_loss=ranking_loss+k.mean(k.square((tmp3*tmp)))
    return ranking_loss

def bootstrap_replicate_1d(data):
    bs_sample = np.random.choice(data, len(data))
    return bs_sample
class loaddata():
    #def __init__(self,time_interval,time_max,time_length):
    def get_y_labels(self,status,time):
        ret = np.ones((status.shape[0], 100))
        for i in range(status.shape[0]):
            if status[i] == 1:
                ret[i, 0:time[i] - 1 + 1] = 0
            elif status[i] == 0:
                ret[i, 0:time[i] + 1] = 0
                ret[i, time[i] + 1:] = 2
        return ret
    def reshape_y(self,y):
        dim = y.shape[1]
        ret = []
        for i in range(dim):
            ret.append(y[:, i, 0:2])
        return ret
    def LoadData(self,df_train,df_test):
        cols_standardize = ['Age', 'AFP', 'Hgb', 'PLT', 'WBC',
                            'AST', 'LDH', 'ALB', 'TBLT',
                            'CRP ', 'PT', 'PT Percentage',
                            'Diameter of main tumor',
                            ]
        cols_leave = ['Gender', 'New lesions']
        cols_categorical = ['Location of Lesions', 'No. of intrahepatic lesions', 'Child-Pugh',
                            'Cause and type of Hepatitis', 'CR/PR?', 'Ascites']

        standardize = [([col], StandardScaler()) for col in cols_standardize]
        leave = [(col, None) for col in cols_leave]
        categorical = [(col, OrderedCategoricalLong()) for col in cols_categorical]

        x_mapper = DataFrameMapper(standardize + leave + categorical)
        x_train = x_mapper.fit_transform(df_train).astype('float32')
        x_test = x_mapper.fit_transform(df_test).astype('float32')
        y_train = self.get_y_labels(df_train['Death'], df_train['Overall Survival Time'])
        y_test = self.get_y_labels(df_test['Death'], df_test['Overall Survival Time'])
        y_train = y_train[:, np.arange(time_interval, time_max, time_interval)]
        y_test = y_test[:, np.arange(time_interval, time_max, time_interval)]
        y_train_status = to_categorical(y_train)
        y_test_status = to_categorical(y_test)
        y_train_status = self.reshape_y(y_train_status)
        y_test_status = self.reshape_y(y_test_status)
        y_train = y_train.astype(np.int32)
        y_test = y_test.astype(np.int32)
        y_train_status_f = y_train_status + [y_train]
        y_test_status_f = y_test_status + [y_test]
        get_target = lambda df: (df['Overall Survival Time'].values, df['Death'].values)
        durations_test, events_test = get_target(df_test)

        return x_train,x_test,y_train,y_train_status,y_train_status_f,y_test_status_f,durations_test, events_test

class Mymodel():
    #def __init__(self,time_length):
    def model(self,x_train,time_length):
        input_tensor=Input((x_train.shape[1],))
        x = input_tensor
        x = Dense(24, activation='sigmoid', kernel_regularizer=L1L2(l1=0., l2=0.))(x)
        x = Dense(16, activation='sigmoid', kernel_regularizer=L1L2(l1=0., l2=0.))(x)
        x = Dense(6, activation='sigmoid', kernel_regularizer=L1L2(l1=0., l2=0.))(x)
        # x = Dropout(dropout, name='dropout')(x)

        prepare_list = {}
        for i in range(time_length):
            prepare_list['x' + str(i)] = Dense(2, activation='softmax', kernel_regularizer=L1L2(l1=0., l2=0.),
                                               name='month_' + str(i))(x)

        xx1 = concatenate(list(prepare_list.values()))
        xx2 = Lambda(lambda x: x[:, 1::2], name='ranking')(xx1)

        model = Model(input_tensor, list(prepare_list.values()) + [xx2])

        return model

def bootstrap_metric_newmodel(model,x_train,x_test,y_train,y_train_status,y_train_status_f,
                         y_test_status_f,durations_test, events_test,):
    c_indexes = []
    ibss = []
    for i in range(100):
        print(i)
        train_bs_idx = bootstrap_replicate_1d(np.array(range(x_train.shape[0])))
        X_tr = x_train[train_bs_idx,]
        Y_tr_0 = y_train[train_bs_idx,]

        Y_tr_1 = []
        for i in range(time_length):
            Y_tr_1.append(y_train_status[i][train_bs_idx])
        Y_tr = Y_tr_1 + [Y_tr_0]
        c_index,ibs=trainmodel2(lambda3, lambda4, lr, batch_size,
                   time_length, X_tr, Y_tr, x_test, y_test_status_f, durations_test, events_test)
        c_indexes.append(c_index)
        ibss.append(ibs)
    print(c_indexes)
    print(ibss)


def evaluate_model(model,x_test,durations_test,events_test):
    y_test_status_pred = model.predict(x_test)
    pred = np.array(y_test_status_pred[0:time_length])
    pred_dead = pred[:, :, 1]
    cif1 = pd.DataFrame(pred_dead, np.arange(time_length) + 1)
    ev1 = EvalSurv(1 - cif1, durations_test // time_interval, events_test == 1, censor_surv='km')
    c_index = ev1.concordance_td('antolini')
    ibs = ev1.integrated_brier_score(np.arange(time_length))
    print('C-index: {:.4f}'.format(c_index))
    print('IBS: {:.4f}'.format(ibs))
    return c_index
def evaluate_model2(model,x_test,durations_test,events_test):
    y_test_status_pred = model.predict(x_test)
    pred = np.array(y_test_status_pred[0:time_length])
    pred_dead = pred[:, :, 1]
    cif1 = pd.DataFrame(pred_dead, np.arange(time_length) + 1)
    ev1 = EvalSurv(1 - cif1, durations_test // time_interval, events_test == 1, censor_surv='km')
    c_index = ev1.concordance_td('antolini')
    ibs = ev1.integrated_brier_score(np.arange(time_length))
    print('C-index: {:.4f}'.format(c_index))
    print('IBS: {:.4f}'.format(ibs))
    return c_index,ibs
def trainmodel(
               df_train,
               lambda3,
               lambda4,
               lr,
               batch_size,
               time_length,
               x_train,
               y_train,
               y_train_status,
               y_train_status_f,
               x_test,
               y_test_status_f,
               durations_test,
               events_test):
    losses = {}
    loss_weights = {}
    for i in range(time_length):
        losses['month_' + str(i)] = logloss(lambda3)
        loss_weights['month_' + str(i)] = 1
    losses['ranking'] = rankingloss
    loss_weights['ranking'] = lambda4
    mymodel=Mymodel()
    model=mymodel.model(x_train,time_length)
    model.compile(optimizer=Adam(lr),
                  loss=losses,
                  loss_weights=loss_weights)
    model.fit(x_train, y_train_status_f, epochs=100, validation_data=(x_test, y_test_status_f),
              batch_size=batch_size, shuffle=True, verbose=0,
              callbacks=[
                  ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=0, mode='auto', min_delta=0.001,
                                    cooldown=0, min_lr=0),
                  EarlyStopping(patience=3)])
    c_index=evaluate_model(model,x_test,durations_test,events_test)
    bootstrap_metric_newmodel(model,x_train,x_test,y_train,y_train_status,y_train_status_f,
                         y_test_status_f,durations_test, events_test,
                         )
def trainmodel2(
               lambda3,
               lambda4,
               lr,
               batch_size,
               time_length,
               x_train,
               y_train_status_f,
               x_test,
               y_test_status_f,
               durations_test,
               events_test):
    losses = {}
    loss_weights = {}
    for i in range(time_length):
        losses['month_' + str(i)] = logloss(lambda3)
        loss_weights['month_' + str(i)] = 1
    losses['ranking'] = rankingloss
    loss_weights['ranking'] = lambda4
    mymodel=Mymodel()
    model=mymodel.model(x_train,time_length)
    model.compile(optimizer=Adam(lr),
                  loss=losses,
                  loss_weights=loss_weights)
    model.fit(x_train, y_train_status_f, epochs=100, validation_data=(x_test, y_test_status_f),
              batch_size=batch_size, shuffle=True, verbose=0,
              callbacks=[
                  ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=0, mode='auto', min_delta=0.001,
                                    cooldown=0, min_lr=0),
                  EarlyStopping(patience=3)])
    c_index,ibs=evaluate_model2(model,x_test,durations_test,events_test)
    return c_index,ibs

if __name__=='__main__':
    path=os.getcwd()+'/data/'
    file='derivation.csv'
    hyper='hyper_result.csv'

    file_name=path+file
    file_hyper=path+hyper

    hyper_parameter=pd.read_csv(file_hyper)
    data = pd.read_csv(file_name)

    lambda3 =hyper_parameter.iloc[best,0]
    lambda4 =hyper_parameter.iloc[best,1]
    lr =hyper_parameter.iloc[best,2]
    batch_size =hyper_parameter.iloc[best,3]
    time_interval = 6
    time_max = 99
    time_length = time_max // time_interval

    test_data = data.sample(frac=0.25)
    traindata = data.drop(test_data.index)
    df_train = traindata.reset_index(drop=True)
    df_test = test_data.reset_index(drop=True)
    class_data=loaddata()
    x_train,x_test,y_train,y_train_status,y_train_status_f,y_test_status_f,durations_test, events_test=class_data.LoadData(df_train,df_test)
    trainmodel(df_train,lambda3,lambda4,lr,batch_size,
                   time_length,x_train,y_train,y_train_status,y_train_status_f,x_test,y_test_status_f,durations_test, events_test)
