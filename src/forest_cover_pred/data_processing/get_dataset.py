from multiprocessing.spawn import prepare
from joblib import dump, load
from pathlib import Path
from sklearn.model_selection import train_test_split

import pandas as pd
import click


def prepare_data(csv_path=None, test_size=0.2, random_state=None):
    df = pd.read_csv(csv_path)
    # Drop Id and constant columns
    df.drop(columns=["Id", "Soil_Type7", "Soil_Type15"], inplace=True)
    X = df.drop(columns=['Cover_Type']).values
    y = df["Cover_Type"].values
    data = train_test_split(X, y, test_size=test_size, random_state=random_state)
    # save prepared data for later
    dump(data, 'data/processed/data.joblib')
    return data


def get_data():
    try:
        data = load('data/processed/data.joblib')
    except FileNotFoundError:
        data = prepare_data()
    return data

if __name__=="__main__":
    get_data()