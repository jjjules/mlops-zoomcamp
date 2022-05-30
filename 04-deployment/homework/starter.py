#!/usr/bin/env python
# coding: utf-8

import os
import sys
import pickle
import pandas as pd

sklearn_version = os.popen('pip freeze | grep scikit-learn').read().strip('\n')
assert sklearn_version == 'scikit-learn==1.0.2', ('Scikt-learn must be installed '
                                                'with version 1.0.2 to ensure compatibility')

CATEGORICAL = ['PUlocationID', 'DOlocationID']
MODEL_FILENAME = 'model.bin'

def load_model(model_filename=MODEL_FILENAME):
    with open('model.bin', 'rb') as f_in:
        dv, lr = pickle.load(f_in)

    return dv, lr

def read_data(filename):
    print(f'Using file {filename}.')
    df = pd.read_parquet(filename)
    
    df['duration'] = df.dropOff_datetime - df.pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[CATEGORICAL] = df[CATEGORICAL].fillna(-1).astype('int').astype('str')
    
    return df


if __name__ == '__main__':
    assert len(sys.argv) == 3, f'You must provide the year and month for the predictions ({len(sys.argv)-1} args provided).'
    try:
        int(sys.argv[2])
    except ValueError:
        print('The month must be a number (e.g. 2 for February).')

    year = int(sys.argv[1])
    month = int(sys.argv[2])
    print(f'Loading data for {year:04d}/{month:02d}...')
    df = read_data(f'https://nyc-tlc.s3.amazonaws.com/trip+data/fhv_tripdata_{year:04d}-{month:02d}.parquet')

    dicts = df[CATEGORICAL].to_dict(orient='records')
    print('Loading model...')
    dv, lr = load_model()
    print('Creating predictions...')
    X_val = dv.transform(dicts)
    y_pred = lr.predict(X_val)
    print(f'First 10 predicted durations: {y_pred[:10]}.')
    print(f'Mean predicted duration for {year:04d}/{month:02d}: {y_pred.mean():.2f}')

    df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')
    # df_result = pd.DataFrame({'pred': y_pred, 'ride_id': df.ride_id})
    # output_file = 'df_result.parquet'
    # df_result.to_parquet(
    #     output_file,
    #     engine='pyarrow',
    #     compression=None,
    #     index=False
    # )