import pandas as pd
import numpy as np
from datetime import datetime

ddf = pd.read_csv("/Users/estherzhang/Downloads/sales_2017-2.csv")
# property = pd.read_csv("/Users/estherzhang/Downloads/sku_property-2.csv")
# skulist_1 = property['sku'].unique()
# cate = property['category'].unique()


charger = pd.read_csv("/Users/estherzhang/Downloads/charger.csv",header=None)
old_name = charger.columns
new_name = ['sku','unit','category','date','cate3']
charger.rename(columns=dict(zip(old_name, new_name)), inplace=True)
skulist = pd.DataFrame(charger['sku'].unique())
old = skulist.columns
new = ['sku']
skulist.rename(columns=dict(zip(old, new)), inplace=True)


temp = pd.merge(skulist, ddf, how="left")
list = pd.DataFrame(temp['sku'].unique())

list.to_csv("/Users/estherzhang/Downloads/charger_list.csv",header=True,encoding='utf-8')



# print(len(skulist))
# print(len(skulist_1))
# print(len(cate))



