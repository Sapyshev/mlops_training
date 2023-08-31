#!/usr/bin/env python
import numpy as np
import pandas as pd
import mlflow
import os

def read_df(filename: str) -> pd.DataFrame:
    """Read data into dataframe"""
    df = pd.read_csv(filename, index_col=0)
    df['vector'] = df['vector'].apply(convert)
    return df

def convert(item):
    item = item.strip()  # remove spaces at the end
    item = item[1:-1]    # remove `[ ]`
    item = np.fromstring(item, sep=' ')  # convert string to `numpy.array`
    return item

def load_model(run_id):
    logged_model = f's3://mlflow-artifacts-remote-rollan/2/{run_id}/artifacts/model'
    model = mlflow.pyfunc.load_model(logged_model)
    return model

def apply_model(input_file, run_id, output_file):
    print(f'reading the data from {input_file}...')
    df = read_df(input_file)
    X = np.stack(df['vector'])

    print(f'loading the model from {run_id}...')
    model = load_model(run_id)
    y_pred = model.predict(X)

    print(f'Apply the results to {output_file}...')
    df['predict'] = y_pred
    df['version'] = run_id
    df.to_parquet(output_file, index=False)

def run():
    input_file = '../../data/valid_data.csv'
    output_file = 'output/rf_predict.parquet'
    RUN_ID = os.getenv('RUN_ID',"7f2ba2ea16c449a88f0e61b087f13fc4")

    apply_model(
        input_file=input_file, 
        run_id=RUN_ID, 
        output_file=output_file)

if __name__ == '__main__':
    run()