# download https://www.kaggle.com/asraful70/talkingdata-added-new-features-in-lightgbm?scriptVersionId=3331854

"""
If you find this kernel helpful please upvote. Also any suggestion for improvement will be warmly welcomed.
I made cosmetic changes in the [code](https://www.kaggle.com/aharless/kaggle-runnable-version-of-baris-kanber-s-lightgbm/code). 
Added some new features. Ran for 25mil chunk rows.
Also taken ideas from various public kernels.
"""

FILENO= 5 #To distinguish the output file name.
debug=0  #Whethere or not in debuging mode

import pandas as pd
import time
import numpy as np
from sklearn.model_selection import train_test_split
import lightgbm as lgb
import gc
#import matplotlib.pyplot as plt
from multiprocessing import cpu_count
import os
import pickle

from logging import getLogger
from logging.config import fileConfig

fileConfig('../conf/logging.conf', defaults={'logpath': '../logs/train.log'})
logger = getLogger(__name__)

###### Feature extraction ######

#### Extracting next click feature 
    ### Taken help from https://www.kaggle.com/nanomathias/feature-engineering-importance-testing
    ###Did some Cosmetic changes 
predictors=[]

def do_datetime(df):
    logger.info('Extracting new datetime features...')

    df['hour'] = pd.to_datetime(df.click_time).dt.hour.astype('int8')
    df['day'] = pd.to_datetime(df.click_time).dt.day.astype('int8')
    df['minute'] = pd.to_datetime(df.click_time).dt.minute.astype('int8')
    df['second'] = pd.to_datetime(df.click_time).dt.second.astype('int8')
    df['dayofweek'] = pd.to_datetime(df.click_time).dt.dayofweek.astype('int8')

    return df


def do_next_Click( df, group_cols, frm_to, agg_type='float32'):
    agg_suffix = 'nextClick'
    logger.info(f">> Extracting {agg_suffix} time calculation features...")

    # Name of new feature
    new_feature = '{}_{}'.format('_'.join(group_cols),agg_suffix)    
    agg_path = '../features/{}/{}.pkl'.format(frm_to, new_feature)

    # Unique list of features to select
    all_features = group_cols + ['click_time']

    # Run calculation
    logger.info(f">> Grouping by {group_cols}, and saving time to {agg_suffix} in: {new_feature}")
    if os.path.exists(agg_path):
        with open(agg_path, 'rb') as f:
            df[new_feature] = pickle.load(f)
    else:
        df[new_feature] = (df[all_features].groupby(group_cols)\
            .click_time.shift(-1) - df.click_time).dt.seconds.astype(agg_type)
        with open(agg_path, 'wb') as f:
            pickle.dump(df[new_feature], f)
    predictors.append(new_feature)
    gc.collect()

    return (df)


def do_prev_Click( df, group_cols, frm_to, agg_type='float32'):
    agg_suffix = 'prevClick'
    logger.info(f">> Extracting {agg_suffix} time calculation features...")

    # Name of new feature
    new_feature = '{}_{}'.format('_'.join(group_cols),agg_suffix)    
    agg_path = '../features/{}/{}.pkl'.format(frm_to, new_feature)

    # Unique list of features to select
    all_features = group_cols + ['click_time']

    # Run calculation
    logger.info(f">> Grouping by {group_cols}, and saving time to {agg_suffix} in: {new_feature}")
    if os.path.exists(agg_path):
        with open(agg_path, 'rb') as f:
            df[new_feature] = pickle.load(f)
    else:
        df[new_feature] = (df.click_time - df[all_features].groupby(group_cols)\
                .click_time.shift(+1) ).dt.seconds.astype(agg_type)
        with open(agg_path, 'wb') as f:
            pickle.dump(df[new_feature], f)
    predictors.append(new_feature)
    gc.collect()

    return (df)


## Below a function is written to extract count feature by aggregating different cols
def do_count( df, group_cols, frm_to, agg_type='uint32', show_max=False, show_agg=True ):
    agg_name='{}_count'.format('_'.join(group_cols))
    agg_path='../features/{}/{}.pkl'.format(frm_to, agg_name)
    if show_agg:
        logger.info( "Aggregating by {} ... and saved in {}".format(group_cols, agg_name))
    if os.path.exists(agg_path):
        with open(agg_path, 'rb') as f:
            gp = pickle.load(f)
    else:
        gp = df[group_cols][group_cols].groupby(group_cols).size().rename(agg_name).to_frame().reset_index()
        with open(agg_path, 'wb') as f:
            pickle.dump(gp, f)
    df = df.merge(gp, on=group_cols, how='left')
    del gp
    if show_max:
        logger.info("{}: max value = {}".format(agg_name, df[agg_name].max()))
    df[agg_name] = df[agg_name].astype(agg_type)
    predictors.append(agg_name)
#     logger.info('predictors',predictors)
    gc.collect()
    return( df )
    
##  Below a function is written to extract unique count feature from different cols
def do_countuniq( df, group_cols, counted, frm_to, agg_type='uint16', show_max=False, show_agg=True ):
    agg_name= '{}_by_{}_countuniq'.format(('_'.join(group_cols)),(counted))  
    agg_path='../features/{}/{}.pkl'.format(frm_to, agg_name)
    if show_agg:
        logger.info( "Counting unqiue {} by {} ... and saved in {}".format(counted, group_cols, agg_name))
    if os.path.exists(agg_path):
        with open(agg_path, 'rb') as f:
            gp = pickle.load(f)
    else:
        gp = df[group_cols+[counted]].groupby(group_cols)[counted].nunique().reset_index().rename(columns={counted:agg_name})
        with open(agg_path, 'wb') as f:
            pickle.dump(gp, f)
    df = df.merge(gp, on=group_cols, how='left')
    del gp
    if show_max:
        logger.info("{}: max value = {}".format(agg_name, df[agg_name].max()))
    df[agg_name] = df[agg_name].astype(agg_type)
    predictors.append(agg_name)
#     logger.info('predictors',predictors)
    gc.collect()
    return( df )
### Below a function is written to extract cumulative count feature  from different cols    
def do_cumcount( df, group_cols, counted, frm_to, agg_type='uint16', show_max=False, show_agg=True ):
    agg_name= '{}_by_{}_cumcount'.format(('_'.join(group_cols)),(counted)) 
    agg_path='../features/{}/{}.pkl'.format(frm_to, agg_name)
    if show_agg:
        logger.info( "Cumulative count by {} ... and saved in {}".format(group_cols, agg_name))
    if os.path.exists(agg_path):
        with open(agg_path, 'rb') as f:
            gp = pickle.load(f)
    else:
        gp = df[group_cols+[counted]].groupby(group_cols)[counted].cumcount()
        with open(agg_path, 'wb') as f:
            pickle.dump(gp, f)
    df[agg_name]=gp.values
    del gp
    if show_max:
        logger.info("{}: max value = {}".format(agg_name, df[agg_name].max()))
    df[agg_name] = df[agg_name].astype(agg_type)
    predictors.append(agg_name)
#     logger.info('predictors',predictors)
    gc.collect()
    return( df )
### Below a function is written to extract mean feature  from different cols
def do_mean( df, group_cols, counted, frm_to, agg_type='float32', show_max=False, show_agg=True ):
    agg_name= '{}_by_{}_mean'.format(('_'.join(group_cols)),(counted))  
    agg_path='../features/{}/{}.pkl'.format(frm_to, agg_name)
    if show_agg:
        logger.info("Calculating mean of {} by {} ... and saved in {}".format(counted, group_cols, agg_name))
    if os.path.exists(agg_path):
        with open(agg_path, 'rb') as f:
            gp = pickle.load(f)
    else:
        gp = df[group_cols+[counted]].groupby(group_cols)[counted].mean().reset_index().rename(columns={counted:agg_name})
        with open(agg_path, 'wb') as f:
            pickle.dump(gp, f)
    df = df.merge(gp, on=group_cols, how='left')
    del gp
    if show_max:
        logger.info("{}: max value = {}".format(agg_name, df[agg_name].max()))
    df[agg_name] = df[agg_name].astype(agg_type)
    predictors.append(agg_name)
#     logger.info('predictors',predictors)
    gc.collect()
    return( df )

def do_var( df, group_cols, counted, frm_to, agg_type='float32', show_max=False, show_agg=True ):
    agg_name= '{}_by_{}_var'.format(('_'.join(group_cols)),(counted)) 
    agg_path='../features/{}/{}.pkl'.format(frm_to, agg_name)
    if show_agg:
        logger.info("Calculating variance of {} by {} ... and saved in {}".format(counted, group_cols, agg_name))
    if os.path.exists(agg_path):
        with open(agg_path, 'rb') as f:
            gp = pickle.load(f)
    else:
        gp = df[group_cols+[counted]].groupby(group_cols)[counted].var().reset_index().rename(columns={counted:agg_name})
        with open(agg_path, 'wb') as f:
            pickle.dump(gp, f)
    df = df.merge(gp, on=group_cols, how='left')
    del gp
    if show_max:
        logger.info("{}: max value = {}".format(agg_name, df[agg_name].max()))
    df[agg_name] = df[agg_name].astype(agg_type)
    predictors.append(agg_name)
#     logger.info('predictors',predictors)
    gc.collect()
    return( df )

###  A function is written to train the lightGBM model with different given parameters
if debug:
    logger.info('*** debug parameter set: this is a test run for debugging purposes ***')

def lgb_modelfit_nocv(params, dtrain, dvalid, predictors, target='target', objective='binary', metrics='auc',
                 feval=None, early_stopping_rounds=50, num_boost_round=3000, verbose_eval=10, categorical_features=None):
    nthread = int(cpu_count() * 0.75)
    lgb_params = {
        'boosting_type': 'gbdt',
        'objective': objective,
        'metric':metrics,
        'learning_rate': 0.05,
        #'is_unbalance': 'true',  #because training data is unbalance (replaced with scale_pos_weight)
        'num_leaves': 31,  # we should let it be smaller than 2^(max_depth)
        'max_depth': -1,  # -1 means no limit
        'min_child_samples': 20,  # Minimum number of data need in a child(min_data_in_leaf)
        'max_bin': 255,  # Number of bucketed bin for feature values
        'subsample': 0.6,  # Subsample ratio of the training instance.
        'subsample_freq': 0,  # frequence of subsample, <=0 means no enable
        'colsample_bytree': 0.3,  # Subsample ratio of columns when constructing each tree.
        'min_child_weight': 5,  # Minimum sum of instance weight(hessian) needed in a child(leaf)
        'subsample_for_bin': 200000,  # Number of samples for constructing bin
        'min_split_gain': 0,  # lambda_l1, lambda_l2 and min_gain_to_split to regularization
        'reg_alpha': 0,  # L1 regularization term on weights
        'reg_lambda': 0,  # L2 regularization term on weights
        'nthread': nthread,
        'verbose': 0,
    }

    lgb_params.update(params)

    logger.info("preparing validation datasets")

    xgtrain = lgb.Dataset(dtrain[predictors].values, label=dtrain[target].values,
                          feature_name=predictors,
                          categorical_feature=categorical_features
                          )
    del dtrain
    gc.collect()

    xgvalid = lgb.Dataset(dvalid[predictors].values, label=dvalid[target].values,
                          feature_name=predictors,
                          categorical_feature=categorical_features
                          )

    del dvalid
    gc.collect()

    evals_results = {}

    bst1 = lgb.train(lgb_params, 
                     xgtrain, 
                     valid_sets=[xgtrain, xgvalid], 
                     valid_names=['train','valid'], 
                     evals_result=evals_results, 
                     num_boost_round=num_boost_round,
                     early_stopping_rounds=early_stopping_rounds,
                     verbose_eval=10, 
                     feval=feval)
    score = evals_results['valid'][metrics][bst1.best_iteration-1]

    logger.info("Model Report")
    logger.info("bst1.best_iteration: {}".format(bst1.best_iteration))
    logger.info("{}: {}".format(metrics, score))

    return (bst1, bst1.best_iteration, score)
    
## Running the full calculation.

#### A function is written here to run the full calculation with defined parameters.

def DO(frm,to,fileno,use_all_agg=True):
    dtypes = {
            'ip'            : 'uint32',
            'app'           : 'uint16',
            'device'        : 'uint8',
            'os'            : 'uint16',
            'channel'       : 'uint16',
            'is_attributed' : 'uint8',
            'click_id'      : 'uint32',
            }

    logger.info('loading train data... {} {}'.format(frm,to))
    train_df = pd.read_csv("../input/train.csv", parse_dates=['click_time'], skiprows=range(1,frm), nrows=to-frm, dtype=dtypes, usecols=['ip','app','device','os', 'channel', 'click_time', 'is_attributed'])

    logger.info('loading test data...')
    if debug:
        test_df = pd.read_csv("../input/test.csv", nrows=100000, parse_dates=['click_time'], dtype=dtypes, usecols=['ip','app','device','os', 'channel', 'click_time', 'click_id'])
    else:
        test_df = pd.read_csv("../input/test.csv", parse_dates=['click_time'], dtype=dtypes, usecols=['ip','app','device','os', 'channel', 'click_time', 'click_id'])

    len_train = len(train_df)
    train_df=train_df.append(test_df)
    
    del test_df
        
    gc.collect()

    train_df = do_datetime(train_df)

    frm_to = '{}_{}'.format(frm, to)
    os.makedirs('../features/{}'.format(frm_to), exist_ok=True)

    train_df = do_next_Click(train_df, ['ip', 'app', 'device', 'os', 'channel'], frm_to); gc.collect()
    train_df = do_next_Click(train_df, ['ip', 'os', 'device'], frm_to); gc.collect()
    train_df = do_next_Click(train_df, ['ip', 'os', 'app', 'device'], frm_to); gc.collect()
    train_df = do_next_Click(train_df, ['device','channel'], frm_to); gc.collect()
    train_df = do_next_Click(train_df, ['app', 'device', 'channel'], frm_to); gc.collect()
    #train_df = do_next_Click(train_df, ['device'], frm_to); gc.collect()
    #train_df = do_next_Click(train_df, ['device', 'hour'], frm_to); gc.collect()

    #train_df = do_prev_Click(train_df, ['ip', 'app', 'device', 'os', 'channel'], frm_to); gc.collect()
    #train_df = do_prev_Click(train_df, ['ip', 'os', 'device'], frm_to); gc.collect()
    #train_df = do_prev_Click(train_df, ['ip', 'os', 'app', 'device'], frm_to); gc.collect()
    #train_df = do_prev_Click(train_df, ['device','channel'], frm_to); gc.collect()
    #train_df = do_prev_Click(train_df, ['app', 'device', 'channel'], frm_to); gc.collect()

    train_df = do_cumcount( train_df, ['ip'], 'os', frm_to); gc.collect()
    train_df = do_cumcount( train_df, ['ip', 'device', 'os'], 'app', frm_to); gc.collect()

    if use_all_agg:
        frm_to = 'all'

    train_df = do_countuniq( train_df, ['ip'], 'channel', frm_to); gc.collect()
    train_df = do_countuniq( train_df, ['ip', 'device', 'os'], 'app', frm_to); gc.collect()
    train_df = do_countuniq( train_df, ['ip', 'day'], 'hour', frm_to); gc.collect()
    train_df = do_countuniq( train_df, ['ip'], 'app', frm_to); gc.collect()
    train_df = do_countuniq( train_df, ['ip', 'app'], 'os', frm_to); gc.collect()
    train_df = do_countuniq( train_df, ['ip'], 'device', frm_to); gc.collect()
    train_df = do_countuniq( train_df, ['app'], 'channel', frm_to); gc.collect()
    train_df = do_count( train_df, ['ip', 'day', 'hour'], frm_to); gc.collect()
    train_df = do_count( train_df, ['ip', 'app'], frm_to); gc.collect()
    train_df = do_count( train_df, ['ip', 'app', 'os'], frm_to); gc.collect()
    train_df = do_count( train_df, ['ip', 'channel'], frm_to); gc.collect()
    train_df = do_count( train_df, ['ip', 'device', 'os', 'app'], frm_to); gc.collect()
    train_df = do_count( train_df, ['ip', 'device'], frm_to); gc.collect()
    train_df = do_count( train_df, ['app', 'channel'], frm_to); gc.collect()
    train_df = do_var( train_df, ['ip', 'day', 'channel'], 'hour', frm_to); gc.collect()
    train_df = do_var( train_df, ['ip', 'app', 'os'], 'hour', frm_to); gc.collect()
    train_df = do_var( train_df, ['ip', 'app', 'channel'], 'day', frm_to); gc.collect()
#    train_df = do_mean( train_df, ['ip', 'app', 'channel'], 'hour', frm_to); gc.collect()
#    train_df = do_mean( train_df, ['ip', 'app', 'os'], 'hour', frm_to); gc.collect()
#    train_df = do_mean( train_df, ['ip', 'app', 'channel'], 'day', frm_to); gc.collect()
    
    logger.debug(train_df.head(5))
    gc.collect()
    
    
    logger.info('Before appending predictors... {}'.format(sorted(predictors)))
    target = 'is_attributed'
    word= ['app','device','os', 'channel', 'hour', 'day','minute', 'second', 'dayofweek']
    for feature in word:
        if feature not in predictors:
            predictors.append(feature)
    categorical = ['app', 'device', 'os', 'channel', 'hour', 'day','minute', 'second', 'dayofweek']
    logger.info('After appending predictors... {}'.format(sorted(predictors)))

    test_df = train_df[len_train:]
    val_df = train_df[(len_train-val_size):len_train]
    train_df = train_df[:(len_train-val_size)]

    logger.info("train size: {}".format(len(train_df)))
    logger.info("valid size: {}".format(len(val_df)))
    logger.info("test size : {}".format(len(test_df)))

    with open('../features/test_df.pkl', 'wb') as f:
        pickle.dump(test_df, f)
    del test_df
    gc.collect()

    logger.info("Training...")
    start_time = time.time()

    params = {
        'learning_rate': 0.1,
        #'is_unbalance': 'true', # replaced with scale_pos_weight argument
        'num_leaves': 31,  # 2^max_depth - 1
        'max_depth': 5,  # -1 means no limit
        'min_child_samples': 20,  # Minimum number of data need in a child(min_data_in_leaf)
        'max_bin': 100,  # Number of bucketed bin for feature values
        'subsample': 0.8,  # Subsample ratio of the training instance.
        'subsample_freq': 1,  # frequence of subsample, <=0 means no enable
        'colsample_bytree': 0.3,  # Subsample ratio of columns when constructing each tree.
        'min_child_weight': 0,  # Minimum sum of instance weight(hessian) needed in a child(leaf)
        'scale_pos_weight':200, # because training data is extremely unbalanced 
        'reg_alpha': 0.1
    }

    search_colsample_bytree = [0.3]
    search_reg_alpha = [1]
    search_max_depth = [5]
    bst = None
    best_params = None
    best_iteration = 0
    best_score = -1

    for cb in search_colsample_bytree:
        for ra in search_reg_alpha:
            for md in search_max_depth:
                params['colsample_bytree'] = cb
                params['reg_alpha'] = ra
                params['max_depth'] = md
                params['num_leaves'] = 2 ** md -1
                (bst1, best_iteration1, auc) = lgb_modelfit_nocv(params, 
                                        train_df, 
                                        val_df, 
                                        predictors, 
                                        target, 
                                        objective='binary', 
                                        metrics='auc',
                                        early_stopping_rounds=50, 
                                        verbose_eval=True, 
                                        num_boost_round=5000, 
                                        categorical_features=categorical)
                logger.info('+ colsample_bytree={}, reg_alpha={}, max_depth={}, : auc = {}'.format(cb, ra, md, auc))
                if auc > best_score:
                    bst = bst1
                    best_params = params
                    best_iteration = best_iteration1
                    best_score = auc
                logger.info('+ current_best_score={}, params={}'.format(best_score, best_params))

    logger.info('[{}]: model training time'.format(time.time() - start_time))
    del train_df
    del val_df
    gc.collect()

    imp = zip(bst.feature_name(), bst.feature_importance(importance_type='split'))
    imp = sorted(imp, key=lambda x:-x[1])
    logger.info('feature importances(split): {}'.format(imp))

    imp = zip(bst.feature_name(), bst.feature_importance(importance_type='gain'))
    imp = sorted(imp, key=lambda x:-x[1])
    logger.info('feature importances(gain): {}'.format(imp))

#    ax = lgb.plot_importance(bst, max_num_features=100)
#    plt.show()
#    plt.savefig('foo.png')

    logger.info("Predicting...")
    with open('../features/test_df.pkl', 'rb') as f:
        test_df = pickle.load(f)
    os.remove('../features/test_df.pkl')
    sub = pd.DataFrame()
    sub['click_id'] = test_df['click_id'].astype('int')
    sub['is_attributed'] = bst.predict(test_df[predictors],num_iteration=best_iteration)
#     if not debug:
#         logger.info("writing...")
    sub.to_csv('../output/sub_it%d.csv.gz'%(fileno),index=False,float_format='%.9f',compression='gzip')
    logger.info("done...")
    return sub
    
    
####### Chunk size defining and final run  ############

nrows=184903891-1
frm=nrows-110000000
nchunk=100000000
val_size=20000000
use_all_agg = True

# use all train data
frm=0
nchunk=nrows

if debug:
    frm=0
    nchunk=100000
    val_size=10000

to=frm+nchunk

sub=DO(frm,to,FILENO,use_all_agg)
