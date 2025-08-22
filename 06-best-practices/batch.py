#!/usr/bin/env python
# coding: utf-8
import os
import pickle
import pandas as pd


def main(year: int, month: int, input_file: str, output_file: str):
    with open('model.bin', 'rb') as f_in:
        dv, lr = pickle.load(f_in)


    categorical = ['PULocationID', 'DOLocationID']

    print(input_file)
    df = read_data(input_file)
    df = prepare_data(df, categorical)
    df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')


    dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(dicts)
    y_pred = lr.predict(X_val)


    print('predicted mean duration:', y_pred.mean())


    df_result = pd.DataFrame()
    df_result['ride_id'] = df['ride_id']
    df_result['predicted_duration'] = y_pred

    print(output_file)
    save_data(df_result, output_file)


def read_data(filename: str) -> pd.DataFrame:
    s3_endpoint_url = os.getenv('S3_ENDPOINT_URL')
    if s3_endpoint_url and filename.startswith('s3://'):
        df = pd.read_parquet(filename, storage_options={
            "client_kwargs": {"endpoint_url": s3_endpoint_url}}
        )
    else:
        df = pd.read_parquet(filename)
    return df


def save_data(df: pd.DataFrame, filename: str):
    s3_endpoint_url = os.getenv('S3_ENDPOINT_URL')
    if s3_endpoint_url and filename.startswith('s3://'):
        df.to_parquet(filename, engine='pyarrow', compression=None, index=False, storage_options={
            "client_kwargs": {"endpoint_url": s3_endpoint_url}}
        )
    else:
        df.to_parquet(filename, engine='pyarrow', compression=None, index=False)


def prepare_data(df: pd.DataFrame, categorical: list) -> pd.DataFrame:
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype(int)

    return df


def get_input_path(year, month):
    default_input_pattern = 's3://nyc-duration/sample_data.parquet'
    input_pattern = os.getenv('INPUT_FILE_PATTERN', default_input_pattern)
    return input_pattern.format(year=year, month=month)


def get_output_path(year, month):
    default_output_pattern = 's3://nyc-duration/predictions.parquet'
    output_pattern = os.getenv('OUTPUT_FILE_PATTERN', default_output_pattern)
    return output_pattern.format(year=year, month=month)


if __name__ == "__main__":
    year = 2023
    month = 1

    input_file = get_input_path(year, month)
    output_file = get_output_path(year, month)

    main(year, month, input_file, output_file)
