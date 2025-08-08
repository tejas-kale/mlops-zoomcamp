import pandas as pd
import pickle
import argparse


def read_data(filename, categorical):
    df = pd.read_parquet(filename)
    
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    
    return df


def main(year: int, month: int):
    categorical = ['PULocationID', 'DOLocationID']

    df = read_data(
        f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year:04d}-{month:02d}.parquet',
        categorical
    )

    with open('model.bin', 'rb') as f_in:
        dv, model = pickle.load(f_in)

    dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(dicts)
    y_pred = model.predict(X_val)

    print(f"Mean predicted duration: {y_pred.mean()}")

    # Save predictions
    if 'ride_id' in df.columns:
        df_result = df["ride_id"].to_frame()
        df_result = df_result.assign(y_pred=y_pred.tolist())
        output_file = f'./yellow_tripdata_{year:04d}-{month:02d}_pred.parquet'
        df_result.to_parquet(
            output_file,
            engine='pyarrow',
            compression=None,
            index=False
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict taxi ride duration.")
    parser.add_argument('--year', type=int, required=True, help='Year of the data.')
    parser.add_argument('--month', type=int, required=True, help='Month of the data.')
    args = parser.parse_args()
    main(args.year, args.month)



