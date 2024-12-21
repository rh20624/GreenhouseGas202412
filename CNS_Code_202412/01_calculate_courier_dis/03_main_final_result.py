import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import math
import matplotlib
import matplotlib.pyplot as plt
import statsmodels.api as sm
import os
import xgboost as xgb

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score, KFold, StratifiedShuffleSplit

pd.options.mode.chained_assignment = None


if __name__ == '__main__':
    df_city_pop = pd.read_csv('../city_level.csv')

    month_list = ['2023_01','2023_07','2024_01']
    for month in month_list:
        print(month)
        df_result_city = pd.DataFrame()
        count = 0
        for root, dirs, files in os.walk(os.path.join('./dis_result',month)):
            for file in files:
                if file[-3:] == 'csv' and 'checkpoint' not in file:
                    print(file)
                    count += 1
                    city = file.split('_')[0]
                    df_temp = pd.read_csv(os.path.join(root,file))
                    df_temp['city'] = city
                    if len(df_result_city) == 0:
                        df_result_city = df_temp
                    else:
                        df_result_city = pd.concat([df_result_city, df_temp], axis=0)
        print(len(df_result_city))
        print(count)
        df_wave = pd.read_csv('jd_site_wave.csv')
        df_wave = df_wave[['site_id','wave_count']]
        df_result_city = pd.merge(df_result_city,df_wave,how='left',on='site_id')
        df_result_city['wave_count'] = df_result_city['wave_count'].fillna(2)
        print(len(df_result_city))
        df_result_city.to_csv(f'./final_result_20240909/JD_final_result_{month}.csv',index=False)
