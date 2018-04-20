import pandas as pd
import xgboost as xgb

from logging import getLogger
from logging.config import fileConfig

fileConfig('../conf/logging.conf', defaults={'logpath': '../logs/train.log'})
logger = getLogger(__name__)


TRAIN_CSV_PATH = '../input/train.csv'
TEST_CSV_PATH = '../input/test.csv'
SAMPLE_SUBMIT_CSV_PATH = '../input/sample_submission.csv'
SUBMIT_CSV_PATH = '../output/submit_xgb.csv'


def load_data(csv_path):
    df = pd.read_csv(csv_path)
    df['click_hour'] = pd.to_datetime(df['click_time']).dt.hour
    df.drop(columns=['click_time'], inplace=True)
    return df


def load_train_data():
    df = load_data(TRAIN_CSV_PATH)
    df.drop(columns=['attributed_time'], inplace=True)
    X = df.drop('is_attributed', axis=1)
    y = df['is_attributed'].values
    return X, y


def load_test_data():
    df = load_data(TEST_CSV_PATH).sort_values('click_id')
    df.drop(columns=['click_id'], inplace=True)
    return df


def main():
    logger.info('main start')

    logger.info('load_train_data() start')
    X_train, y_train = load_train_data()
    logger.info('load_train_data() end')

    params = {
        'max_depth': 3,
        'min_child_weight': 10,
        'gamma': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'colsample_bylevel': 0.8,
        'reg_alpha': 0.1,
        'learning_rate': 0.1,
        'n_estimators': 1000,
        'n_jobs': 16
    }

    # train model
    logger.info('train start')
    clf = xgb.XGBClassifier(**params)
    clf.fit(X_train, y_train)
    logger.info('train end')

    # predict
    logger.info('load_test_data() start')
    X_test = load_test_data()
    logger.info('load_test_data() end')
    logger.info('predict start')
    probs = clf.predict_proba(X_test)[:, 1]
    logger.info('predict end')

    # submit file
    df_submit = pd.read_csv(SAMPLE_SUBMIT_CSV_PATH).sort_values('click_id')
    df_submit['is_attributed'] = probs
    df_submit.to_csv(SUBMIT_CSV_PATH, index=False)

    logger.info('main end')


if __name__ == '__main__':
    main()
