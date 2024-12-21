import os
import re
import json
import math
import datetime
import time
import warnings
warnings.filterwarnings('ignore')
from math import ceil
import glob
import geopandas as gpd
import pandas as pd
import numpy as np

from geopy.distance import distance

filtered_df = pd.read_csv('jd_site_wave.csv')
wave_city_dict = {}
month = '2023_01'


def longest_common_substring_length(s1, s2):
    len1, len2 = len(s1), len(s2)
    dp = [[0] * (len2 + 1) for _ in range(len1 + 1)]    
    max_length = 0     
    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
                max_length = max(max_length, dp[i][j])
            else:
                dp[i][j] = 0
    return max_length 


def find_aoi_file(source_path, file_origin):
    city_origin = file_origin.split('_')[0]
    for root, dirs, files in os.walk(source_path):
        for file in files:
            city_now = file.split('_')[0]
            if city_now == city_origin:
                print(f'success:{file}')
                df = pd.read_csv(os.path.join(root, file))
                return df
    for root, dirs, files in os.walk(source_path):
        for file in files:
            city_now = file.split('_')[0]
            max_length = longest_common_substring_length(city_origin,city_now)
            min_origin_length = min(len(city_origin),len(city_now))
            if max_length/min_origin_length > 0.5:
                print(f'success:{file}')
                df = pd.read_csv(os.path.join(root, file))
                return df
                    
    print(f'{file_origin} error')
    return None


def initial_wave_city_dict():
    filtered_city_list = filtered_df['city'].unique().tolist()
    for root, dirs, files in os.walk(f'./middle_results/{month}'):
        for file in files:
            city_origin = file.split('_')[0]
            flag = True
            for filtered_city in filtered_city_list:
                if filtered_city == city_origin:
                    wave_city_dict[city_origin] = filtered_city
                    print(f'{city_origin}:{filtered_city}')
                    flag = False
                    break
            if flag:
                for filtered_city in filtered_city_list:
                    max_length = longest_common_substring_length(city_origin,filtered_city)
                    min_origin_length = min(len(city_origin),len(filtered_city))
                    if max_length/min_origin_length > 0.5:
                        wave_city_dict[city_origin] = filtered_city
                        print(f'{city_origin}:{filtered_city}')
                        flag = False
                        break
            if flag:
                print(f'{city_origin} error')


def cal_travel_distance(city):
    try:
        tj_orders = pd.read_csv(f'middle_results/{month}/{city}_orders_final_{month}.csv')
        tj_orders.rename(columns={'site_name':'sitename','delivery_time':'deliverytime'},inplace=True)
        df_aoi_info = find_aoi_file('../AOI_all_city/', f'{city}_AOI.csv')
        print(df_aoi_info.shape)
        df_aoi_info['aoi_polygon'] = gpd.GeoSeries.from_wkt(df_aoi_info['aoiWKT'])
        df_aoi_info['polygon_centroid'] = df_aoi_info['aoi_polygon'].apply(lambda x: x.centroid)
        df_aoi_info['polygon_center_lng'] = df_aoi_info['polygon_centroid'].apply(lambda cid: cid.x)
        df_aoi_info['polygon_center_lat'] = df_aoi_info['polygon_centroid'].apply(lambda cid: cid.y)

        random_coords = tj_orders.groupby('sitename').apply(lambda x: x.sample(1)).reset_index(drop=True)
        merged_df_bj = tj_orders.merge(random_coords[['sitename', 'lat', 'lng']], on='sitename', suffixes=('', '_site'))
        merged_df_bj = merged_df_bj.rename(columns={'lat_site': 'site_lat', 'lng_site': 'site_lng'})
        print(merged_df_bj.shape)
        sitename_list = filtered_df[filtered_df['city']==wave_city_dict[city]]['sitename'].unique().tolist()
        df_trace_seq = merged_df_bj[merged_df_bj['sitename'].isin(sitename_list)]
        print(df_trace_seq.shape)
        df_trace_seq['deliverytime'] = pd.to_datetime(df_trace_seq['deliverytime'])
        time_length = pd.Timedelta(minutes=10)
        df_trace_seq['time_index_30'] = (df_trace_seq['deliverytime'].dt.hour * 60 + df_trace_seq['deliverytime'].dt.minute) * 60 // time_length.total_seconds() 
        df_trace_seq['time_index_30'] = df_trace_seq['time_index_30'].astype(int)
        df_trace_seq['minute_index'] = df_trace_seq['deliverytime'].dt.hour * 60 + df_trace_seq['deliverytime'].dt.minute
        print(df_trace_seq.shape)
        df_trace_seq.rename(columns={'aois': 'aoiId'}, inplace=True)
        
        df_trace_seq_gb = df_trace_seq.groupby(by=['operator_id', 'sitename', 'site_lat', 'site_lng', 'time_index_30', 'aoiId'], as_index=False).apply(
        lambda x: pd.Series({
            'waybill_code': x['waybill_code'].count(),
            'max_minute_index': x['minute_index'].max(),
            'min_minute_index': x['minute_index'].min(),
            'max_minute_index_5_percent': x['minute_index'].quantile(0.95),
            'min_minute_index_5_percent': x['minute_index'].quantile(0.05)
        }))
        print(df_trace_seq_gb.shape)
        df_trace_seq_gb.sort_values(by=['operator_id', 'sitename', 'site_lat', 'site_lng', 'time_index_30', 'waybill_code'], ascending=[True, True, True, True, True, False], inplace=True)
        print(df_trace_seq_gb.shape)
        df_trace_seq_gb.drop_duplicates(subset=['operator_id', 'sitename', 'site_lat', 'site_lng', 'time_index_30'], inplace=True)
        print(df_trace_seq_gb.shape)

        df_trace_seq_gb_stay = df_trace_seq_gb.copy()
        df_trace_seq_gb_stay.sort_values(by=['operator_id', 'sitename', 'site_lat', 'site_lng', 'time_index_30'], inplace=True)

        df_trace_seq_gb_stay['flag'] = (df_trace_seq_gb_stay['aoiId'] != df_trace_seq_gb_stay['aoiId'].shift(1)).cumsum()

        df_trace_seq_gb_stay_updown = df_trace_seq_gb_stay.groupby(
            ['operator_id', 'sitename', 'site_lat', 'site_lng', 'flag'], as_index=False
        ).agg({'aoiId': 'first', 'min_minute_index': 'first', 'max_minute_index': 'last', 'min_minute_index_5_percent': 'first', 'max_minute_index_5_percent': 'last'})

        df_trace_seq_gb_stay_updown['stay_distance'] = (df_trace_seq_gb_stay_updown['max_minute_index'] - df_trace_seq_gb_stay_updown['min_minute_index']) / 60 * 4
        df_trace_seq_gb_stay_updown['stay_distance_5_percent'] = (df_trace_seq_gb_stay_updown['max_minute_index_5_percent'] - df_trace_seq_gb_stay_updown['min_minute_index_5_percent']) / 60 * 4
        df_trace_seq_gb_stay_updown.loc[df_trace_seq_gb_stay_updown.stay_distance_5_percent<0, 'stay_distance_5_percent'] = 0
        print(df_trace_seq_gb_stay_updown.shape)

        df_trace_seq_gb_stay_updown_gb = df_trace_seq_gb_stay_updown.groupby(['operator_id', 'sitename', 'site_lat', 'site_lng'], as_index=False)[['stay_distance', 'stay_distance_5_percent']].sum()
        print(df_trace_seq_gb_stay_updown_gb.shape)

        #-------------------------------Distance between AOI------------------------------------
        def cal_distance(lat1, lng1, lat2, lng2):
            if (lat1 is None) or (lng1 is None) or (lat2 is None) or (lng2 is None):
                return 0
            return distance((lat1, lng1), (lat2, lng2)).miles * 1.60934
        
        df_trace_seq_gb_trace = df_trace_seq_gb.merge(df_aoi_info[['aoiId', 'polygon_center_lat', 'polygon_center_lng']], on=['aoiId'], how='inner')
        print(df_trace_seq_gb_trace.shape)
        df_trace_seq_gb_trace['shift_lat'] = df_trace_seq_gb_trace.groupby(by=['operator_id', 'sitename', 'site_lat', 'site_lng'])['polygon_center_lat'].shift()
        df_trace_seq_gb_trace['shift_lng'] = df_trace_seq_gb_trace.groupby(by=['operator_id', 'sitename', 'site_lat', 'site_lng'])['polygon_center_lng'].shift()
        df_trace_seq_gb_trace['aoi_distance'] = 0
        df_trace_seq_gb_trace.loc[
            (df_trace_seq_gb_trace['polygon_center_lat'].notnull())&
            (df_trace_seq_gb_trace['polygon_center_lng'].notnull())&
            (df_trace_seq_gb_trace['shift_lat'].notnull())&
            (df_trace_seq_gb_trace['shift_lng'].notnull()),
            'aoi_distance'
        ] = df_trace_seq_gb_trace.loc[
            (df_trace_seq_gb_trace['polygon_center_lat'].notnull())&
            (df_trace_seq_gb_trace['polygon_center_lng'].notnull())&
            (df_trace_seq_gb_trace['shift_lat'].notnull())&
            (df_trace_seq_gb_trace['shift_lng'].notnull())
        ].apply(lambda row: cal_distance(row['polygon_center_lat'], row['polygon_center_lng'], row['shift_lat'], row['shift_lng']), axis=1)

        df_trace_seq_gb_trace_gb = df_trace_seq_gb_trace.groupby(by=['operator_id', 'sitename'], as_index=False)['aoi_distance'].sum()
        print(df_trace_seq_gb_trace_gb.shape)

        #-------------------------------AOI Distance from site------------------------------------
        df_trace_seq_gb_trace_site = df_trace_seq_gb_trace.copy()
        df_trace_seq_gb_trace_site['min_time_index'] = df_trace_seq_gb_trace_site.groupby(by=['operator_id', 'sitename', 'site_lat', 'site_lng'])['time_index_30'].transform('min')
        df_trace_seq_gb_trace_site['max_time_index'] = df_trace_seq_gb_trace_site.groupby(by=['operator_id', 'sitename', 'site_lat', 'site_lng'])['time_index_30'].transform('max')
        df_trace_seq_gb_trace_site = df_trace_seq_gb_trace_site[
            (df_trace_seq_gb_trace_site['min_time_index']==df_trace_seq_gb_trace_site['time_index_30']) | 
            (df_trace_seq_gb_trace_site['max_time_index']==df_trace_seq_gb_trace_site['time_index_30'])
        ]
        df_trace_seq_gb_trace_site = df_trace_seq_gb_trace_site.drop(['shift_lat', 'shift_lng', 'aoi_distance', 'min_time_index', 'max_time_index'], axis=1)
        print(df_trace_seq_gb_trace_site.shape)
        df_trace_seq_gb_trace_site['site_distance'] = df_trace_seq_gb_trace_site.apply(lambda row: cal_distance(row['polygon_center_lat'], row['polygon_center_lng'], row['site_lat'], row['site_lng']), axis=1)

        station_wave_target_city = filtered_df[filtered_df['city'] == city]
        df_trace_seq_gb_trace_site_gb = df_trace_seq_gb_trace_site.groupby(by=['operator_id', 'sitename'], as_index=False)['site_distance'].sum()
        print(df_trace_seq_gb_trace_site_gb.shape)

        df_trace_seq_gb_trace_site_gb = pd.merge(df_trace_seq_gb_trace_site_gb, station_wave_target_city, on='sitename', how='left')
        df_trace_seq_gb_trace_site_gb = df_trace_seq_gb_trace_site_gb.rename(columns={'wave_count': 'cishu'})
        print(df_trace_seq_gb_trace_site_gb.shape)
        df_trace_seq_gb_trace_site_gb['cishu'] = df_trace_seq_gb_trace_site_gb['cishu'].fillna(0)
        df_trace_seq_gb_trace_site_gb['site_distance_all'] = df_trace_seq_gb_trace_site_gb['site_distance'] * df_trace_seq_gb_trace_site_gb['cishu']

        #--------------------------------Distance merging-----------------------------------
        df_sf_distance = df_trace_seq_gb_trace_gb.merge(df_trace_seq_gb_trace_site_gb.drop(['site_distance', 'cishu'], axis=1), on=['operator_id', 'sitename'], how='outer')
        df_sf_distance = df_sf_distance.merge(df_trace_seq_gb_stay_updown_gb.drop(['site_lng', 'site_lat'], axis=1), on=['operator_id', 'sitename'], how='outer')
        df_sf_distance['aoi_distance'] = df_sf_distance['aoi_distance'].fillna(0)
        df_sf_distance['site_distance_all'] = df_sf_distance['site_distance_all'].fillna(0)
        df_sf_distance['stay_distance'] = df_sf_distance['stay_distance'].fillna(0)
        df_sf_distance['stay_distance_5_percent'] = df_sf_distance['stay_distance_5_percent'].fillna(0)
        df_sf_distance['all_distance'] = df_sf_distance['aoi_distance'] + df_sf_distance['site_distance_all'] + df_sf_distance['stay_distance']
        print(df_sf_distance.shape)
        df_sf_distance = df_sf_distance[df_sf_distance.all_distance>0]
        print(df_sf_distance.shape)
    except Exception as e:
        print(f'{city}:error')
        print(f"error: {e}")
        return 
    
    return df_sf_distance


if __name__ == '__main__':
    print(month)
    print('----------------------------------------------------------------------------------')
    initial_wave_city_dict()
    print('----------------------------------------------------------------------------------')
    if not os.path.exists(f'./dis_result/{month}'):
        os.makedirs(f'./dis_result/{month}')
    for root, dirs, files in os.walk(f'./middle_results/{month}'):
        for file in files:
            file_save = file.replace('orders_final','dis')
            city = file.split('_')[0]
            if not os.path.exists(os.path.join(f'./dis_result/{month}',file_save)):
                print(city)
                df_result = cal_travel_distance(city)
                if isinstance(df_result, pd.DataFrame):
                    df_result.to_csv(os.path.join(f'./dis_result/{month}',file_save),index=False)
            else:
                print(f'{city} dis_result is exist')

    for root, dirs, files in os.walk(f'./middle_results/{month}'):
        for file in files:
            if 'checkpoint' in file:
                os.remove(os.path.join(root,file))