from genericpath import exists
from pathlib import Path
from joblib import dump
import click

from ..data_processing.get_dataset import get_data
from .pipeline import create_pipe


@click.command()
@click.option("-d", "--dataset-path", 
    default='data/raw/train.csv', 
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    show_default=True)
@click.option("-s", "--save-model-path", 
    default='models/model.joblib',
    type=click.Path(dir_okay=False, writable=True, path_type=Path),
    show_default=True)
@click.option("--test-split-ratio", 
    default=0.2, 
    type=click.FloatRange(0, 1, min_open=True, max_open=True),
    show_default=True)
@click.option("--random-state", 
    default=42,
    show_default=True)
def train(dataset_path, save_model_path, test_split_ratio, random_state):
    X_tr, X_val, y_tr, y_val = get_data(dataset_path, test_split_ratio, random_state)
    pipe = create_pipe()
    pipe.fit(X_tr, y_tr)
    click.echo("model trained")
    click.echo(f"train acc={pipe.score(X_tr, y_tr)} test_acc={pipe.score(X_val, y_val)}")
    dump(pipe, save_model_path)
    click.echo(f"model saved to {save_model_path}")