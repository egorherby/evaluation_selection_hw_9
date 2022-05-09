from genericpath import exists
from pathlib import Path
import random
from joblib import dump
import click

from ..data_processing.get_dataset import get_data
from .pipeline import create_pipe

from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV, cross_val_score, StratifiedKFold


@click.command()
@click.option(
    "-d",
    "--dataset-path",
    default="data/raw/train.csv",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    show_default=True,
)
@click.option(
    "-s",
    "--save-model-path",
    default="models/model.joblib",
    type=click.Path(dir_okay=False, writable=True, path_type=Path),
    show_default=True,
)
@click.option(
    "--test-split-ratio",
    default=0.2,
    type=click.FloatRange(0, 1, min_open=True, max_open=True),
    show_default=True,
)
@click.option("--random-state", default=42, show_default=True)
@click.option("--metric", default='accuracy', show_default=True,
    help="other metrics are `f1_macro` or `roc_auc_ovr`")
def train(dataset_path, save_model_path, metric, test_split_ratio, random_state):
    X_tr, X_val, y_tr, y_val = get_data(dataset_path, test_split_ratio, random_state)
    pipe = create_pipe()

    param_grid = {
        "tree__criterion":["gini", "entropy"],
        "tree__splitter":["best", "random"],
        "tree__max_depth":[None, 3,5,7,10,15,20,30,50],
        "tree__min_samples_leaf":[1, 3, 5, 7, 10, 20, 25, 30],
        "tree__max_features":["sqrt","log2",None]
    }
    outer_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=random_state)
    inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    clf = RandomizedSearchCV(pipe, param_grid, cv=inner_cv, n_iter=100, n_jobs=-1, scoring=metric)
    nested_score = cross_val_score(clf, X_tr, y_tr, n_jobs=-1, scoring=metric)
    click.echo(f"nested scores {metric}={nested_score.mean()}")
    # pipe.fit(X_tr, y_tr)
    click.echo("model trained")
    # click.echo(
    #     f"train acc={pipe.score(X_tr, y_tr)} test_acc={pipe.score(X_val, y_val)}"
    # )
    # dump(pipe, save_model_path)
    # click.echo(f"model saved to {save_model_path}")
