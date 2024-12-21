import pandas as pd
import os
import shutil

from tqdm import tqdm
from ast import literal_eval
from multiprocessing import Pool
from shapely.vectorized import contains
from shapely.wkt import loads


def cal_iterows(sites,sel_ord,sel_aoi,month):
    orders_lng = sel_ord.lng.tolist()
    orders_lat = sel_ord.lat.tolist()
    for index,row in sel_aoi.iterrows():
        flag = contains(row['aoiWKT'], orders_lng, orders_lat)
        sel_ord.loc[flag,'aois']=row['aoiId']
    sel_ord.to_csv(f'out/{month}/{sites}.csv',index=False)


def find_aoi_file(source_path, file_origin):
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
    

if __name__ == '__main__':
    month = '2023_01'
    merged_order_addr_gis = pd.read_csv(f'./order_addr_gis/JD_order_gis_{month}.csv',
                                        encoding='gbk')
    merged_order_addr_gis.rename(columns={'city_name': 'city'}, inplace=True)
    for city in ['Beijing','Tianjing','Chongqing','Shanghai']:
        merged_order_addr_gis.loc[merged_order_addr_gis['city'] == city, 'city'] = city + '市'
    orders = merged_order_addr_gis
    city_list = orders['city'].unique()

    directory_path = os.path.join('out',month)
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

    for city in city_list:

        print(city)
        if os.path.isfile(f'middle_results/{month}/{city}_orders_final_{month}.csv'):
            print(f'{city} is exist')
            continue
        
        try:
            orders_city = orders[orders['city'] == city]
            orders_city['sites']=''
            sites = find_aoi_file('../AOI_station_all_city/', f'{city}_station.csv')
            sites.aoiid = sites.aoiid.map(lambda x:literal_eval(x))

            orders_lng = orders_city.lng.tolist()
            orders_lat = orders_city.lat.tolist()
            t = tqdm(sites.iterrows(),total=(len(sites)))
            for index,row in t:
                flag = contains(loads(row['polygon']), orders_lng, orders_lat)
                orders_city.loc[flag,'sites']=row['siteid']
            t.close()

            orders_tj = orders_city
            orders_tj['aois']=''

            aoi = find_aoi_file('../AOI_all_city/', f'{city}_AOI.csv')
            aoi.aoiWKT = aoi.aoiWKT.map(lambda x:loads(x))

            pool = Pool(processes = 24)
            
            for filename in os.listdir(directory_path):
                if filename[-3:] == 'csv':
                    file_path = os.path.join(directory_path, filename)
                    os.unlink(file_path)
            for index,row in sites.iterrows():
                select_orders=orders_tj[orders_tj.sites==row['siteid']].copy()
                select_orders.reset_index(inplace=True,drop=True)
        
                if len(select_orders)==0:continue
                select_aoi=aoi[aoi.aoiId.isin(row['aoiid'])].copy()
                pool.apply_async(cal_iterows,(row['siteid'],select_orders,select_aoi,month))
            pool.close()
            pool.join() 

            files=os.listdir(directory_path)
            if '.ipynb_checkpoints' in files:
                files.remove('.ipynb_checkpoints')

            res=[]
            for i in files:
                filepath=os.path.join(directory_path,i)
                res.append(pd.read_csv(filepath))

            if len(res)>0:
                res=pd.concat(res)
                if not os.path.exists(f'middle_results/{month}'):
                    os.makedirs(f'middle_results/{month}')
                res.to_csv(f'middle_results/{month}/{city}_orders_final_{month}.csv',index=False)
            else:
                print('length is 0')
        except Exception as e:
            print(f'{city}:error')
            print(f'{e}')

