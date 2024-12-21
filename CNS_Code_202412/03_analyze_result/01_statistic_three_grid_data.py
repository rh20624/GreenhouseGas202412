import numpy as np
import pandas as pd
import os
import json


def convert_to_plot_df2(df_final_result):
    df = df_final_result
    new_rows = []
    for i in range(1,5):
        new_row = {'city':f'point{i}'}
        for column in df.columns:
            if column != 'city':
                new_row[column] = 0
        new_rows.append(new_row)
    new_df = pd.DataFrame(new_rows)
    df = pd.concat([df, new_df], ignore_index=True)

    print(df.columns)
    filtered_data1 = df['sum_ce_2023_01'][df['sum_ce_2023_01'] != 0]  
    filtered_data2 = df['sum_ce_2023_07'][df['sum_ce_2023_07'] != 0] 
    filtered_data3 = df['sum_ce_2024_01'][df['sum_ce_2024_01'] != 0] 
    filtered_data = pd.concat([filtered_data1, filtered_data2], axis=0)
    filtered_data = pd.concat([filtered_data, filtered_data3], axis=0)
    quintiles_month = np.percentile(filtered_data, [20, 40, 60, 80])
    
    for column in df.columns:
        if column != 'city':
            filtered_data = df[column][df[column] != 0]
            print(type(filtered_data))
            quintiles = np.percentile(filtered_data, [20, 40, 60, 80])
            if column in ['sum_ce_2023_01','sum_ce_2023_07','sum_ce_2024_01']:
                quintiles = quintiles_month
            for i in range(len(df)):
                if df.loc[i,column] == 0:
                    df.loc[i,column] = -1
                elif df.loc[i,column] < quintiles[0]:
                    df.loc[i,column] = 1
                elif df.loc[i,column] < quintiles[1]:
                    df.loc[i,column] = 2
                elif df.loc[i,column] < quintiles[2]:
                    df.loc[i,column] = 3
                elif df.loc[i,column] < quintiles[3]:
                    df.loc[i,column] = 4
                else:
                    df.loc[i,column] = 5
            for i in range(1,5):
                df.loc[df['city']==f'point{i}', column] = quintiles[i-1]

    return df


def make_plot_20240826():
    df = pd.read_csv('../data/quan_sum_result_20240911.csv')

    '''
    sum_Ce_year: Total carbon emissions per city in a year
    sum_Ce_day: Carbon emissions per city per day
    average_operator_Ce_day: Average carbon emissions per town
    average_package_Ce_day: Average carbon emissions per package
    sum_Ce_2023_01:2023_01 Total carbon emissions per city (sum_Ce_2023_07, sum_Ce_2024_01)
    '''
    
    df['sum_Ce_year'],df['sum_Ce_day'],df['average_operator_Ce_day'],df['average_package_Ce_day'] = 0,0,0,0
    df['sum_Ce_2023_01'],df['sum_Ce_2023_07'],df['sum_Ce_2024_01'] = 0,0,0
    
    company_list = ['JD','SF','YT','YD','ZT','ST']
    month_list = ['2023_01','2023_07','2024_01']
    pkg_num_dict = {}
    for month in month_list:
        pkg_num_dict[month] = {}
        for company in company_list:
            pkg_num_dict[month][company] = df[f'num_package_{company}_{month}'].sum()
    operator_num_dict = {}
    for month in month_list:
        operator_num_dict[month] = {}
        for company in company_list:
            operator_num_dict[month][company] = df[f'num_operator_{company}_{month}'].sum()
    
    columns_list = [] 
    for month in month_list:
        for company in company_list:
            columns_list.append(f'sum_Ce_{company}_{month}')
            df['sum_Ce_day'] += df[f'sum_Ce_{company}_{month}']  
    df['sum_Ce_day'] = df['sum_Ce_day']/3  
    df['sum_Ce_day'] = df['sum_Ce_day']/1000 

    for month in month_list:
        for company in company_list:
            df[f'sum_Ce_{month}'] += df[f'sum_Ce_{company}_{month}']
        df[f'sum_Ce_{month}'] *= 30 
        df[f'sum_Ce_{month}'] /= 1000 

    # The total emission of cities with 0 emission value is set to 0, and some companies are missing
    df['has_zero'] = df[columns_list].eq(0).any(axis=1) 
    df.loc[df['has_zero'], 'sum_Ce_day'] = 0
    for month in month_list:
        df.loc[df['has_zero'], f'sum_Ce_{month}'] = 0
        df.loc[df['has_zero'], 'sum_Ce_2023_01'] = 0
        df.loc[df['has_zero'], 'sum_Ce_2023_07'] = 0
        df.loc[df['has_zero'], 'sum_Ce_2024_01'] = 0
    df['sum_Ce_year'] = 365 * df['sum_Ce_day']
    
    # Calculate the total number of employees in 5 companies on an average day
    df['num_operator_one_day'] = 0
    for month in month_list:
        for company in company_list:
            df['num_operator_one_day'] += df[f'num_operator_{company}_{month}']
    df['num_operator_one_day'] /= 3

    # Calculate the total number of parcels per day on average
    sum_package = 0
    for key, value in pkg_num_dict.items():
        for key2,value2 in value.items():
            sum_package += value2
    sum_package = int(sum_package/3)
    
    for i in range(len(df)):
        city = df.loc[i,'city']
        if df.loc[i,'sum_Ce_day'] > 0:
            df.loc[i, 'average_operator_Ce_day'] =  df.loc[i,'sum_Ce_day']/df.loc[i,'num_operator_one_day']
            num_package = sum_package * df.loc[i,'ratio']
            df.loc[i, 'average_package_Ce_day'] = df.loc[i,'sum_Ce_day'] / num_package
        else:
            df.loc[i, 'average_operator_Ce_day'] = 0
            df.loc[i, 'average_package_Ce_day'] = 0
        
        
    df = df[['city','sum_Ce_day','average_operator_Ce_day','average_package_Ce_day','sum_Ce_2023_01','sum_Ce_2023_07','sum_Ce_2024_01','sum_Ce_year','elevator_ratio_2024_01']]

    # Convert the unit of average_operator to kg and the unit of average_package to g
    for column in df.columns:
        if 'average_operator' in column:
            df[column] = df[column] * 1000
        elif 'average_package' in column:
            df[column] = df[column] * 1000000
    df.to_csv('../data/quan_three_grid_city_level_Ce.csv',index=False)
    df = df.rename(columns=str.lower)
    
    df = convert_to_plot_df2(df)
    df.to_csv('../data/plot_20240924.csv',index=False)


if __name__ == '__main__':
    make_plot_20240826()
    