from joblib import dump, load
from pathlib import Path
from sklearn.model_selection import train_test_split

import pandas as pd
import click


def get_data(csv_path, test_size=0.2, random_state=None):
    processed_path = Path("data/processed/data.joblib")
    try:
        data = load(processed_path)
        click.echo("Processed data was successfully loaded")
    except FileNotFoundError:
        df = pd.read_csv(csv_path)
        click.echo("Data was read from csv file")
        # Drop Id and constant columns
        df.drop(columns=["Id", "Soil_Type7", "Soil_Type15"], inplace=True)
        X = df.drop(columns=["Cover_Type"]).values
        y = df["Cover_Type"].values
        data = train_test_split(X, y, test_size=test_size, random_state=random_state)
        # save prepared data for later
        dump(data, processed_path)
        click.data(f"Processed and saved data to {processed_path} for later use")
    return data


if __name__ == "__main__":
    get_data()
