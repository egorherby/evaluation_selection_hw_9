# from joblib import dump

# import pandas as pd
import click

@click.command()
@click.option("-d","--dataset-path",
    default='data/processed/',
    show_default=True)
def train(dataset_path):
    click.echo(f"data path is {dataset_path}")
    pass