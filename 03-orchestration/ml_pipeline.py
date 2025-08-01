import argparse
import os

import mlflow
import pandas as pd
from prefect import flow, task
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression


@task
def read_data(base_url: str, year: int, month: int) -> pd.DataFrame:
    """
    Reads a Parquet file and returns a DataFrame.

    Args:
        base_url (str): The base URL of the Parquet file to read.
    Returns:
        pd.DataFrame: The DataFrame containing the data from the Parquet file.
    """
    file_path = f"{base_url}/yellow_tripdata_{year}-{month:02d}.parquet"
    return pd.read_parquet(file_path)


@task
def preprocess_data(df_taxi: pd.DataFrame) -> pd.DataFrame:
    """
    Get the duration of the trip in minutes, filter out trips with
    a duration less than 1 minute or greater than 60 minutes,
    and convert pickup and dropoff location IDs to string type.

    Args:
        df (pd.DataFrame): The DataFrame to preprocess.
    Returns:
        pd.DataFrame: The preprocessed DataFrame.
    """
    df_taxi["duration"] = df_taxi.tpep_dropoff_datetime - df_taxi.tpep_pickup_datetime
    df_taxi["duration"] = df_taxi["duration"].dt.total_seconds() / 60

    df_taxi = df_taxi[(df_taxi["duration"] >= 1) & (df_taxi["duration"] <= 60)]

    categorical = ["PULocationID", "DOLocationID"]
    df_taxi[categorical] = df_taxi[categorical].astype(str)

    return df_taxi


@task
def train_model(df_train: pd.DataFrame, features: list, target: str) -> None:
    """
    Train a linear regression model using the provided DataFrame.

    Args:
        df_train (pd.DataFrame): The DataFrame containing training data.
        features (list): List of feature column names.
        target (str): The target column name.
    Returns:
        DictVectorizer, LinearRegression: The fitted DictVectorizer and
            LinearRegression model.
    """
    dv = DictVectorizer()
    train_dicts = df_train[features].to_dict(orient="records")

    X_train = dv.fit_transform(train_dicts)
    y_train = df_train[target].values

    lr = LinearRegression()
    lr.fit(X_train, y_train)

    return dv, lr


@flow
def main(
    base_url: str,
    year: int,
    month: int,
    mlflow_tracking_uri: str,
    experiment_name: str,
) -> pd.DataFrame:
    """
    Main function to read data from a Parquet file.

    Args:
        base_url (str): The base URL of the Parquet file to read.
        year (int): The year of the data to process.
        month (int): The month of the data to process.
        mlflow_tracking_uri (str): The MLflow tracking URI.
        experiment_name (str): The name of the MLflow experiment.

    Returns:
        pd.DataFrame: The DataFrame containing the data from the Parquet file.
    """
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run():
        # Log parameters to MLflow
        mlflow.log_param("year", year)
        mlflow.log_param("month", month)
        mlflow.log_param("base_url", base_url)

        # Execute the pipeline steps
        df_taxi = read_data(base_url, year, month)
        df_taxi = preprocess_data(df_taxi)
        taxi_dv, taxi_lr = train_model(
            df_taxi, features=["PULocationID", "DOLocationID"], target="duration"
        )

        # Log metrics and models to MLflow
        mlflow.log_metric("num_rows", len(df_taxi))
        mlflow.sklearn.log_model(taxi_dv, "dict_vectorizer")
        mlflow.sklearn.log_model(taxi_lr, "linear_regression")
        mlflow.log_artifact(__file__)

        # Log a sample of the DataFrame as a CSV file
        df_taxi.head().to_csv("sample_data.csv", index=False)
        mlflow.log_artifact("sample_data.csv")
        os.remove("sample_data.csv")

        # Register the model in MLflow
        mlflow.register_model(
            "runs:/{}/linear_regression".format(mlflow.active_run().info.run_id),
            "TaxiDurationModel",
        )

        return {"processed_df": df_taxi, "dv": taxi_dv, "model": taxi_lr}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the ML pipeline.")
    parser.add_argument(
        "--base_url",
        type=str,
        default="https://d37ci6vzurychx.cloudfront.net/trip-data",
        help="Base URL of the data",
    )
    parser.add_argument(
        "--year", type=int, default=2023, help="Year of the data to process"
    )
    parser.add_argument(
        "--month", type=int, default=3, help="Month of the data to process"
    )
    parser.add_argument(
        "--mlflow_tracking_uri",
        type=str,
        default="sqlite:///taxi_experiment.db",
        help="MLflow tracking URI",
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        default="taxi_experiment",
        help="MLflow experiment name",
    )

    args = parser.parse_args()
    obj = main(
        base_url=args.base_url,
        year=args.year,
        month=args.month,
        mlflow_tracking_uri=args.mlflow_tracking_uri,
        experiment_name=args.experiment_name,
    )
    print(f"Number of rows: {len(obj['processed_df'])}")
    print(f"Model intercept: {obj['model'].intercept_}")
