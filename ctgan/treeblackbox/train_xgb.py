import torch
import numpy as np
from sklearn.model_selection import train_test_split 
import pandas as pd 
import lightgbm as lgb
import xgboost as xgb

INPUT_FILE_PATH = '/Users/sahanalva/Downloads/application_train.csv'
OUTPUT_MODEL_PATH = '/Users/sahanalva/Counterfactual Research/tree_model_test.dat'

# read the test files 
app_train = pd.read_csv(INPUT_FILE_PATH)


app_train['is_test'] = 0
app_train['is_train'] = 1

# target variable
Y = app_train['TARGET']
train_X = app_train.drop(['TARGET'], axis = 1)
data = train_X

cats = ['NAME_CONTRACT_TYPE',
 'CODE_GENDER',
 'FLAG_OWN_CAR',
 'FLAG_OWN_REALTY',
 'NAME_TYPE_SUITE',
 'NAME_INCOME_TYPE',
 'NAME_EDUCATION_TYPE',
 'NAME_FAMILY_STATUS',
 'NAME_HOUSING_TYPE',
 'OCCUPATION_TYPE',
 'WEEKDAY_APPR_PROCESS_START',
 'ORGANIZATION_TYPE', 
 'CNT_CHILDREN',
 'CNT_FAM_MEMBERS',
 'REG_REGION_NOT_LIVE_REGION']

nums= ['AMT_INCOME_TOTAL',
        'AMT_CREDIT',
        'AMT_ANNUITY',
        'AMT_GOODS_PRICE',
        'REGION_POPULATION_RELATIVE',
        'DAYS_BIRTH',
        'DAYS_EMPLOYED',
        'DAYS_REGISTRATION',
        'DAYS_ID_PUBLISH',
        'EXT_SOURCE_1',
        'EXT_SOURCE_2',
        'EXT_SOURCE_3',
        'DAYS_LAST_PHONE_CHANGE']

meta = ['SK_ID_CURR','is_train','is_test']


def _get_categorical_features(df):
    feats = [col for col in list(df.columns) if df[col].dtype == 'object']
    return feats

# function to factorize categorical features
def _factorize_categoricals(df, cats):
    for col in cats:
        df[col], _ = pd.factorize(df[col])
    return df 

# function to create dummy variables of categorical features
def _get_dummies(df, cats):
    for col in cats:
        df = pd.concat([df, pd.get_dummies(df[col], prefix=col)], axis=1)
    return df 


data = data[meta+cats+nums]
data[nums] = data[nums].fillna(0)
data[cats] = data[cats].fillna("NA")
data = _factorize_categoricals(data, cats)

ignore_features = ['SK_ID_CURR', 'is_train', 'is_test']
relevant_features = [col for col in data.columns if col not in ignore_features]
trainX = data[data['is_train'] == 1][relevant_features]
params = {'max_depth':7,'min_child_weight':1,'eval_metric':'auc','alpha':0.5, 'lambda':0.5, 'objective':'binary:logistic'}
xgb_train = xgb.DMatrix(data = trainX, label=Y)
xgb_eval = xgb.DMatrix(data=x_val, label=y_val)

evallist = [(xgb_train, 'eval'), (xgb_train, 'train')]
num_round = 100
bst = xgb.train(params, xgb_train, num_round,evallist)
pickle.dump(bst, open(OUTPUT_MODEL_PATH, "wb"))