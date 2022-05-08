from joblib import dump
import pandas as pd
from sklearn.model_selection import train_test_split

def make_dataset(path=None, test_size=0.2, random_state=None):
    if path is None:
        path="data/raw/train.csv"
    df = pd.read_csv(path)
    # Drop Id and constant columns
    df.drop(columns=["Id", "Soil_Type7", "Soil_Type15"], inplace=True)
    X = df.drop(columns=['Cover_Type']).values
    y = df["Cover_Type"].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    print('up to here is ok')
    dump((X_train, X_test, y_train, y_test,), 'data/processed/train_data.joblib')