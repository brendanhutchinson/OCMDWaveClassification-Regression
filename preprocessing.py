#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np 
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import plotly.express as px 
import seaborn as sns 
import matplotlib.pyplot as plt 
import plotly.figure_factory as ff


# In[3]:


met2020= pd.read_csv('~/Downloads/standardmeteorogical2020.txt', encoding='utf8', delimiter= '\\s+')
met2018 = pd.read_csv('~/Downloads/44009h2018.txt',encoding='utf8', delimiter= '\\s+')
met2019 = pd.read_csv('~/Downloads/44009h2019.txt',encoding='utf8', delimiter= '\\s+')
met2017 = pd.read_csv('~/Downloads/44009h2017.txt',encoding='utf8', delimiter= '\\s+')


# In[343]:


met2018.columns


# In[344]:


def clean_df(df): 
    '''
    input: dataframe
    params: remove string row, 
            create date column, 
            convert to numeric values, 
            replace 999 inputation with Nan
            filter columns with > 90% missing values 
    output: cleaned dataframe
    '''
    df = df.iloc[1:,:]

    df['date'] = df['#YY'] + '-' +  df['MM'] + '-' + df['DD']
    df['date'] = df['date'].apply(pd.to_datetime)

    df.iloc[:,:18] = df.iloc[:,:18].astype('float')

    df =df.replace(999.0, np.nan)
    df =df.replace(99.0, np.nan)
    df =df.replace(9999.0, np.nan)

    df = df[['#YY', 'MM', 'DD', 'hh', 'mm', 'WDIR', 'WSPD', 'GST', 'WVHT', 'DPD',
       'APD', 'MWD', 'PRES', 'ATMP', 'WTMP','date']]

    return df
    


# In[345]:


met2017 =clean_df(met2017)
met2018 = clean_df(met2018)
met2019 = clean_df(met2019)
met2020 = clean_df(met2020)


# In[346]:

bouy_data = pd.concat([met2017,met2018, met2019,met2020], axis =0)

# In[347]:

## dropping missing values - come back for imputation 

bouy_data.dropna(inplace = True)

bouy_data.to_csv('indianriverbouydata2017-2020.csv')



# In[356]:


# additional Data 

additional_measurements= pd.read_csv('/Users/brendanhutchinson/Downloads/additionalmeasuresments202edited.txt', encoding = 'utf8', delimiter = '\\s+')

spectral_direction = pd.read_csv('/Users/brendanhutchinson/Downloads/spectralwavedirection.txt',  delimiter= '\\s+')

spectral_density = pd.read_csv('/Users/brendanhutchinson/Downloads/sprectralwavedensity.txt', encoding='utf8', delimiter= '\\s+')

wind_direction = pd.read_csv('/Users/brendanhutchinson/Downloads/continouswinds2020.txt', encoding='utf8', delimiter= '\\s+')

