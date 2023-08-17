import mlflow
import pandas as pd
import scipy
from sklearn.utils import shuffle
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import pickle
import numpy as np
import io
import xgboost as xgb
from prefect import flow,task

def convert(item):
    item = item.strip()  # remove spaces at the end
    item = item[1:-1]    # remove `[ ]`
    item = np.fromstring(item, sep=' ')  # convert string to `numpy.array`
    return item

@task(retries=3, retry_delay_seconds=2)
def read_data(filename: str) -> pd.DataFrame:
    """Read data into dataframe"""
    df = pd.read_csv(filename)
    df['vector'] = df['vector'].apply(convert)
    return df

@task
def add_features(df_train, df_val):
    """"""
    X_train = df_train['vector']
    y_train = df_train['label']

    X_val = df_val['vector']
    y_val = df_val['label']

    X_train = np.stack(X_train)
    y_train = np.stack(y_train)

    X_val = np.stack(X_val)
    y_val = np.stack(y_val)
    return X_train, y_train, X_val, y_val

@task(log_prints=True)
def train_best_model(X_train, y_train, X_val, y_val) -> None:
    with mlflow.start_run():
        train = xgb.DMatrix(X_train, label=y_train)
        valid = xgb.DMatrix(X_val, label=y_val)

        best_params = {
            'learning_rate': 0.13998783607276,
            'max_depth': 28,
            'min_child_weight':	0.8037214370553903,
            'objective': 'reg:linear',
            'reg_alpha': 0.007917567259199893,
            'reg_lambda': 0.00982476912100121,
            'seed': 42
        }

        mlflow.log_params(best_params)

        booster = xgb.train(
                params=best_params,
                dtrain=train,
                num_boost_round=30,
                evals=[(valid, "validation")],
                early_stopping_rounds=20)
        y_pred = booster.predict(valid)
        rmse = mean_squared_error(y_val, y_pred, squared=False)
        mlflow.log_metric("rmse", rmse) 
        mlflow.xgboost.log_model(booster, artifact_path="models_mlflow")
    return None

@flow
def main_flow(
        train_path: str = "../data/train_data.csv",
        val_path: str = "../data/valid_data.csv",
) -> None:
    """"""
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("my-fakenews-exp")

    df_train = read_data(train_path)
    df_val = read_data(val_path)

    X_train, y_train, X_val, y_val = add_features(df_train, df_val)

    train_best_model(X_train, y_train, X_val, y_val)

if __name__ == "__main__": main_flow()
