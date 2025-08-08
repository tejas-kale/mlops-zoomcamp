# Homework 2

In the second week's homework, the goal is to get familiar with MLflow.

First, let us load the required packages.


```python
import mlflow
```

## Question 1: Install MLflow

The installed MLflow version is `3.1.1`.


```python
mlflow.__version__
```

## Question 2: Download and preprocess data

There were 4 files saved to the output folder.


```python
!python ./preprocess_data.py --raw_data_path ../data/green_taxi_2023 --dest_path ./output
```

## Question 3: Train a model with autolog

The value of `min_samples_split` is *2*. To enable autolog, a simple `mlflow.autolog()` call is sufficient without needing to explicitly start a run.


```python
!python ./train.py
```

## Question 4: Launch the tracking server locally

The other argument to pass is `default-artifact-root`.


```python
#!mlflow server --backend-store-uri sqlite:///backend.db --default-artifact-root ./artifacts
```

## Question 5: Tune model hyperparameters

The best validation RMSE is `5.335`.


```python
!python ./hpo.py
```

## Question 6: Promote the best model to the model registry

The test RMSE of the best model is `5.567`.


```python
!python ./register_model.py
```


```python

```
