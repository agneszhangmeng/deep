import pandas as pd
import numpy as np
from datetime import datetime


# with open("/Users/estherzhang/Downloads/lny/holiday_model_result.csv") as file:
#     ddf= file.read()

ddf = pd.read_csv("/Users/estherzhang/Downloads/lny/holiday_model_result.csv")
skulist = pd.read_csv("/Users/estherzhang/Downloads/lny/sku_active.csv")
skulist['skuseq'] = skulist['skuseq'].map(lambda sku:sku.replace(',',''))
old_name = skulist.columns
new_name = ['sku']
skulist.rename(columns=dict(zip(old_name, new_name)), inplace=True)
skulist['sku'] = skulist['sku'].astype('int')

# unit3 = pd.read_csv("/Users/estherzhang/Downloads/unit3.csv")
# unit3['skuseq'] = unit3['skuseq'].map(lambda sku:sku.replace(',',''))
# old_name = unit3.columns
# new_name = ['sku', 'unit1','unit2', 'cate3']
# unit3.rename(columns=dict(zip(old_name, new_name)), inplace=True)
# unit3['sku'] = unit3['sku'].astype('int')

# ddf = pd.merge(data, unit3, on=['sku','unit1', 'unit2'], how="left")
# ddf = ddf.head(100)

date_range = pd.date_range(start=datetime(2018,1,5), end = datetime(2018,3,5)).strftime('%Y-%m-%d').tolist()
ddf.drop(ddf.columns[3:4],axis=1,inplace=True)
ddf[date_range] = pd.DataFrame([x for x in ddf['mean_list'].apply(lambda x: x.split(","))])
#ddf.drop(ddf.columns[3:4],axis=1,inplace=True)

ddf = pd.merge(skulist, ddf, how="left")
ddf.replace('nan', 0, inplace=True)
ddf.replace('inf', 0, inplace=True)

ddf.iloc[:,3:] = ddf.iloc[:,3:].apply(pd.to_numeric)
ddf['sku_sum'] = ddf.iloc[:,3:].apply(lambda x: x.sum(), axis=1)
new_df = ddf.iloc[:,1:].groupby(['unit1','unit2']).apply(sum)
new_df = new_df.iloc[:,2:]

new_df.to_csv("/Users/estherzhang/Downloads/lny/lny_sum.csv",header=True,encoding='utf-8')

