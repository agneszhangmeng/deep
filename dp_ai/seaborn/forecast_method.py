# - *- coding: utf- 8 - *-

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor

train_data = pd.read_csv("/Users/estherzhang/PycharmProjects/learnpython/seaborn/train.csv", encoding = 'utf-8')
test_data = pd.read_csv("/Users/estherzhang/PycharmProjects/learnpython/seaborn/test.csv", encoding = 'utf-8')

def show_missing(houseprice):
    missing = houseprice.columns[houseprice.isnull().any()].tolist()
    return missing

def cat_exploration(houseprice, column):
    print(houseprice[column].value_counts())

def cat_imputation(houseprice, column, value):
    houseprice.loc[houseprice[column].isnull(), column] = value


#print(test_data['LotFrontage'].corr(test_data['LotArea']))
#print(train_data['LotFrontage'].corr(train_data['LotArea']))

test_data['SqrtLotArea'] = np.sqrt(test_data['LotArea'])
train_data['SqrtLotArea'] = np.sqrt(train_data['LotArea'])

#print(test_data['LotFrontage'].corr(test_data['SqrtLotArea']))
#print(train_data['LotFrontage'].corr(train_data['SqrtLotArea']))

cond = test_data['LotFrontage'].isnull()
test_data.LotFrontage[cond] = test_data.SqrtLotArea[cond]
cond = train_data['LotFrontage'].isnull()
train_data.LotFrontage[cond] = train_data.SqrtLotArea[cond]

del test_data['SqrtLotArea']
del train_data['SqrtLotArea']

#cat_exploration(test_data, 'MSZoning')
#print(test_data[test_data['MSZoning'].isnull() == True])
#print(pd.crosstab(test_data.MSSubClass, test_data.MSZoning))

test_data.loc[test_data['MSSubClass'] == 20, 'MSZoning'] = 'RL'
test_data.loc[test_data['MSSubClass'] == 30, 'MSZoning'] = 'RM'
test_data.loc[test_data['MSSubClass'] == 70, 'MSZoning'] = 'RM'

cat_imputation(test_data, 'Alley', 'None')
cat_imputation(train_data, 'Alley', 'None')

test_data = test_data.drop(['Utilities'], axis=1)
train_data = train_data.drop(['Utilities'], axis=1)

cat_exploration(test_data, 'Exterior1st')
cat_exploration(train_data, 'Exterior1st')
print(test_data[['Exterior1st', 'Exterior2nd']][test_data['Exterior1st'].isnull() == True])
print(pd.crosstab(test_data.Exterior1st, test_data.ExterQual))

test_data.loc[test_data['Exterior1st'].isnull(), 'Exterior1st'] = 'VinylSd'
test_data.loc[test_data['Exterior2nd'].isnull(), 'Exterior2nd'] = 'VinylSd'

basement_cols = ['BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'BsmtFinSF1', 'BsmtFinSF2']
for cols in basement_cols:
    if 'FinFS' not in cols:
        cat_imputation(train_data, cols, 'None')

basement_cols = ['Id', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2']
test_data.loc[test_data['Id'] == 580, 'BsmtCond'] = 'TA'
test_data.loc[test_data['Id'] == 725, 'BsmtCond'] = 'TA'
test_data.loc[test_data['Id'] == 1064, 'BsmtCond'] = 'TA'

for cols in basement_cols:
    if cols not in 'SF' and cols not in 'Bath':
        test_data.loc[test_data['BsmtFinSF1'] == 0.0, cols] = 'None'
for cols in basement_cols:
    if test_data[cols].dtype == np.object:
        cat_imputation(test_data, cols, 'None')
    else:
        cat_imputation(test_data, cols, 0.0)

cat_imputation(test_data, 'BsmtFinSF1', '0')
cat_imputation(test_data, 'BsmtFinSF2', '0')
cat_imputation(test_data, 'BsmtUnfSF', '0')
cat_imputation(test_data, 'TotalBsmtSF', '0')
cat_imputation(test_data, 'BsmtFullBath', '0')
cat_imputation(test_data, 'BsmtHalfBath', '0')

train_data = train_data.replace({'BsmtQual': {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, np.NaN: 0}})
test_data = test_data.replace({'BsmtQual': {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, np.NaN: 0}})

cat_imputation(test_data, 'KitchenQual', 'TA')
cat_imputation(test_data, 'Functional', 'Typ')
cat_imputation(test_data, 'FireplaceQu', 'None')
cat_imputation(train_data, 'FireplaceQu', 'None')

garage_cols = ['GarageType', 'GarageQual', 'GarageCond', 'GarageYrBlt', 'GarageFinish', 'GarageCars', 'GarageArea']
for cols in garage_cols:
    if train_data[cols].dtype == np.object:
        cat_imputation(train_data, cols, 'None')
    else:
        cat_imputation(train_data, cols, 0)

garage_cols = ['GarageType', 'GarageQual', 'GarageCond', 'GarageYrBlt', 'GarageFinish', 'GarageCars', 'GarageArea']

for cols in garage_cols:
    if test_data[cols].dtype == np.object:
        cat_imputation(test_data, cols, 'None')
    else:
        cat_imputation(test_data, cols, 0)

test_data = test_data.drop(['PoolQC'], axis=1)
train_data = train_data.drop(['PoolQC'], axis=1)
test_data = test_data.drop(['PoolArea'], axis=1)
train_data = train_data.drop(['PoolArea'], axis=1)

cat_imputation(test_data, 'Fence', 'None')
cat_imputation(train_data, 'Fence', 'None')

cat_imputation(test_data, 'MiscFeature', 'None')
cat_imputation(train_data, 'MiscFeature', 'None')

cat_imputation(test_data, 'SaleType', 'WD')
cat_imputation(train_data, 'Electrical', 'SBrkr')

train_data = train_data.replace({"MSSubClass": {20: "A", 30: "B", 40: "C", 45: "D", 50: "E",
                                                60: "F", 70: "G", 75: "H", 80: "I", 85: "J",
                                                90: "K", 120: "L", 150: "M", 160: "N", 180: "O", 190: "P"}})
test_data = test_data.replace({"MSSubClass": {20: "A", 30: "B", 40: "C", 45: "D", 50: "E",
                                              60: "F", 70: "G", 75: "H", 80: "I", 85: "J",
                                              90: "K", 120: "L", 150: "M", 160: "N", 180: "O", 190: "P"}})

c = ['BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath']
for cols in c:
    tmp_col = test_data[cols].astype(pd.np.float64)
    tmp_col = pd.DataFrame({cols: tmp_col})
    del test_data[cols]
    test_data = pd.concat((test_data, tmp_col), axis=1)

for cols in train_data.columns:
    if train_data[cols].dtype == np.object:
        train_data = pd.concat((train_data, pd.get_dummies(train_data[cols], prefix=cols)), axis=1)
        del train_data[cols]

for cols in test_data.columns:
    if test_data[cols].dtype == np.object:
        test_data = pd.concat((test_data, pd.get_dummies(test_data[cols], prefix=cols)), axis=1)
        del test_data[cols]

col_train = train_data.columns
col_test = test_data.columns
for index in col_train:
    if index in col_test:
        pass
    else:
        del train_data[index]

col_train = train_data.columns
col_test = test_data.columns
for index in col_test:
    if index in col_train:
        pass
    else:
        del test_data[index]

print(train_data.columns)
#start model
# etr = RandomForestRegressor(n_estimators=400)
# train_y = train_data['SalePrice']
# train_x = train_data.drop(['SalePrice', 'Id'], axis=1)
# etr.fit(train_x, train_y)
# print(etr.feature_importances_)

# imp = etr.feature_importances_
# imp = pd.DataFrame({'feature': train_x.columns, 'score': imp})
# print(imp.sort(['score'], ascending=[0]))  # 按照特征重要性, 进行降序排列, 最重要的特征在最前面
# imp = imp.sort(['score'], ascending=[0])
# imp.to_csv("../feature_importances2.csv", index=False)