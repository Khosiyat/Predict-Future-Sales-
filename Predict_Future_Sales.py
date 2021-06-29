import pandas as pd                                      
import numpy as np                    
from scipy import optimize, stats 
import os
from pathlib import Path


# visualisation libraries
import matplotlib.pyplot as plt                      
import seaborn as sns                

# algorithmic library
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold

from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Flatten, Dropout
from tensorflow.keras.layers import LeakyReLU, PReLU, ELU


root_dir = Path('/kaggle/')
dir_ = root_dir / 'input/competitive-data-science-predict-future-sales'
list(dir_.glob('*'))

#read files
df_train = pd.read_csv(dir_ / 'sales_train.csv')
df_items = pd.read_csv(dir_ / 'items.csv')
df_item_categories = pd.read_csv(dir_ / 'item_categories.csv')
df_shops = pd.read_csv(dir_ / 'shops.csv')
df_test = pd.read_csv(dir_ / 'test.csv')


def featuresetsExtraction_step( tsts_df, trn_df):

    trn_df.drop(['date_block_num','item_price'], axis=1, inplace=True)
    trn_df['date'] = pd.to_datetime(trn_df['date'], dayfirst=True)
    trn_df['date'] = trn_df['date'].apply(lambda x: x.strftime('%Y-%m'))
    main_dataFrame = trn_df.groupby(['date','shop_id','item_id']).sum()
    main_dataFrame = main_dataFrame.pivot_table(index=['shop_id','item_id'], columns='date', values='item_cnt_day', fill_value=0)
    main_dataFrame.reset_index(inplace=True)
    tsts_df = pd.merge(tsts_df, main_dataFrame, on=['shop_id','item_id'], how='left')
    tsts_df.drop(['ID', '2013-01'], axis=1, inplace=True)
    tsts_df = tsts_df.fillna(0)
    trn_y = main_dataFrame['2015-10'].values
    trn_x = main_dataFrame.drop(['2015-10'], axis = 1)
    X_test = tsts_df
    x_train, x_test, y_train, y_test = train_test_split( trn_x, trn_y, test_size=0.2, random_state=101)
    LR = LinearRegression()
    LR.fit(x_train,y_train)
    print('\n', 'Mean Square Error(trainSet):', mean_squared_error(y_train, LR.predict(x_train)), '\n', 'Mean Square Error(testSet):', mean_squared_error(y_test, LR.predict(x_test)), '\n','\n', 'SCORE(testSet):', LR.score(x_train,y_train))
    submission = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/sample_submission.csv') 
    submission.to_csv('PredictFutureSales.csv', index=False)  
featuresetsExtraction_step(df_test,df_train )


