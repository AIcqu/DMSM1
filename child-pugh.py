#child pugh

import pandas as pd
import numpy as np
import random


path = os.getcwd()+'/data/'
file  = 'derivationraw.csv'
file_name = path+file
data = pd.read_csv(file_name)

#Chigh pugh
for i in range(0,data.shape[0]):
    data.loc[i,'Child-Pugh']+=1
    if data['Ascites'][i]==0:
        data.loc[i,'Child-Pugh']+=1
    else:
        if data['Ascites'][i]==1 or data['Ascites'][i]==2:
            data.loc[i,'Child-Pugh']+=2
        else:
            data.loc[i,'Child-Pugh']+=3
    if data['TBLT(umol/L)'][i]<34:
        data.loc[i,'Child-Pugh']+=1
    else:
        if data['TBLT(umol/L)'][i]>=34 and data['TBLT(umol/L)'][i]<=51:
            data.loc[i,'Child-Pugh']+=2
        else:
            data.loc[i,'Child-Pugh']+=3
    if data['ALB (g/L)'][i]>35:
        data.loc[i,'Child-Pugh']+=1
    else:
        if data['ALB (g/L)'][i]>=28 and data['ALB (g/L)'][i]<=35:
            data.loc[i,'Child-Pugh']+=2
        else:
            data.loc[i,'Child-Pugh']+=3
    if data['PT (seconds)'][i]<4:
        data.loc[i,'Child-Pugh']+=1
    else:
        if data['PT (seconds)'][i]>=4 and data['PT (seconds)'][i]<=6:
            data.loc[i,'Child-Pugh']+=2
        else:
            data.loc[i,'Child-Pugh']+=3
data.drop(data[data['Child-Pugh']>9].index,inplace=True)