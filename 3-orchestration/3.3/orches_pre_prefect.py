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
from prefect import flow,task

def convert(item):
    item = item.strip()  # remove spaces at the end
    item = item[1:-1]    # remove `[ ]`
    item = np.fromstring(item, sep=' ')  # convert string to `numpy.array`
    return item

def read_data(filename: str) -> pd.DataFrame:
    """Read data into dataframe"""
    df = pd.read_csv(filename)
    df['vector'] = df['vector'].apply(convert)
    return df

def add_features(
        df_train: pd.DataFrame,
        df_val: pd.DataFrame
) -> tuple(
    [
        scipy.sparse._csr.csr_matrix,
        scipy.sparse._csr.csr_matrix,
        np.ndarray,
        np.ndarray
    ]
):
    X_train = df_train['vector']
    y_train = df_train['label']

    X_val = df_val['vector']
    y_val = df_val['label']

    X_train = np.stack(X_train)
    y_train = np.stack(y_train)

    X_val = np.stack(X_val)
    y_val = np.stack(y_val)







def train_best_model(
)


def main_flow(
        
)