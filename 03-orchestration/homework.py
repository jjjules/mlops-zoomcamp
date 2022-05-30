import pickle
import pandas as pd
import datetime
from dateutil.relativedelta import relativedelta

from prefect import flow, task
from prefect import get_run_logger

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

@task
def get_paths(date, logger):
    date = datetime.datetime.strptime(date, '%Y-%m-%d')
    train_date = date - relativedelta(months=+2)
    
    # Assuming we stay in year 2021
    train_month = train_date.month
    valid_month = train_month + 1
    
    path_template = 'data/fhv_tripdata_2021-{:02d}.parquet'
    train_path = path_template.format(train_month)
    valid_path = path_template.format(valid_month)
    logger.info(f'Train data used: {train_path}')
    logger.info(f'Validation data used: {valid_path}')
    
    return train_path, valid_path

@task
def read_data(path):
    df = pd.read_parquet(path)
    return df

@task
def prepare_features(df, categorical, logger, train=True):
    df['duration'] = df.dropOff_datetime - df.pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60
    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    mean_duration = df.duration.mean()
    if train:
        logger.info(f"The mean duration of training is {mean_duration}")
    else:
        logger.info(f"The mean duration of validation is {mean_duration}")
    
    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    return df

@task
def train_model(df, categorical, logger):

    train_dicts = df[categorical].to_dict(orient='records')
    dv = DictVectorizer()
    X_train = dv.fit_transform(train_dicts) 
    y_train = df.duration.values

    logger.info(f"The shape of X_train is {X_train.shape}")
    logger.info(f"The DictVectorizer has {len(dv.feature_names_)} features")

    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_train)
    mse = mean_squared_error(y_train, y_pred, squared=False)
    logger.info(f"The MSE of training is: {mse}")
    return lr, dv

@task
def run_model(df, categorical, dv, lr, logger):
    val_dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(val_dicts) 
    y_pred = lr.predict(X_val)
    y_val = df.duration.values

    mse = mean_squared_error(y_val, y_pred, squared=False)
    logger.info(f"The MSE of validation is: {mse}")
    return

@flow
def main(date=None):
    if date is None:
        date = str(datetime.date.today())
    assert isinstance(date, str), 'The date should be a string representation of the date as YYYY-MM-DD or None.'
        
    prefect_logger = get_run_logger()
    prefect_logger.info(f'Running flow with date={date}')

    categorical = ['PUlocationID', 'DOlocationID']
    
    train_path, val_path = get_paths(date, prefect_logger).result()
    
    df_train = read_data(train_path)
    df_train_processed = prepare_features(df_train, categorical, prefect_logger)

    df_val = read_data(val_path)
    df_val_processed = prepare_features(df_val, categorical, prefect_logger, train=False)

    # train the model
    lr, dv = train_model(df_train_processed, categorical, prefect_logger).result()
    with open(f'models/lr-{date}.pkl', 'wb') as fmodel, open(f'models/dv-{date}.pkl', 'wb') as fvectorizer:
        pickle.dump(lr, fmodel)
        pickle.dump(dv, fvectorizer)
    run_model(df_val_processed, categorical, dv, lr, prefect_logger)

from prefect.deployments import DeploymentSpec
from prefect.orion.schemas.schedules import CronSchedule
from prefect.flow_runners import SubprocessFlowRunner
DeploymentSpec(
    flow=main,
    name="model_training",
    schedule=CronSchedule(cron='0 9 15 * *'),
    flow_runner=SubprocessFlowRunner()
)
