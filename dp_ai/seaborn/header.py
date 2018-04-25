import pandas as pd
import csv
from datetime import datetime

workSpace = "/Users/estherzhang/Downloads/change_uplift/"
up_7 = pd.read_csv(workSpace + "uplift_result-original.csv", sep=',')
up_7.rename(columns=lambda x: x.strip(), inplace=True)

mapping = pd.read_csv(workSpace + "mapping.csv")
mapping = mapping.drop_duplicates()
up = pd.DataFrame()
# sku	unitname1	unitname2	catecode3	cate3	catecode4	cate4	catecode5	cate5	startDate	endDate	uplift	reason	owner
head = up_7.columns
head = head.str.strip()
titleList = list(head)

for cell in titleList[1:]:
    tmp = pd.DataFrame()
    unit = mapping[mapping['unitname2'] == cell]['unitname1']
    tmp['sku'] = ''
    tmp['catecode3'] = ''
    tmp['cate3'] = ''
    tmp['catecode4'] = ''
    tmp['cate4'] = ''
    tmp['catecode5'] = ''
    tmp['cate5'] = ''
    tmp['uplift'] = up_7[cell]
    tmp['uplift'] = tmp['uplift'].map('{:.1%}'.format)
    tmp['reason'] = 'Holiday Model'
    tmp['owner'] = 'esther'
    tmp['unitname2'] = cell
    tmp['unitname1'] = unit.iloc[0]
    listname = pd.to_datetime(up_7[titleList[0]],format = '%d/%m/%Y')
    tmp['startdate'] = listname
    tmp['enddate'] = listname
    cols_name = list(tmp)
    cols_name.insert(1, cols_name.pop(cols_name.index('unitname1')))
    cols_name.insert(2, cols_name.pop(cols_name.index('unitname2')))
    cols_name.insert(9, cols_name.pop(cols_name.index('startdate')))
    cols_name.insert(10, cols_name.pop(cols_name.index('enddate')))
    tmp = tmp.ix[:, cols_name]
    up = up.append(tmp)
up.to_csv(workSpace + 'dashboard_uplift_input.csv', index=None)
