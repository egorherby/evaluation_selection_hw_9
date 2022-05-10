from pathlib import Path
from joblib import dump
import click
import mlflow

from ..data_processing.get_dataset import get_data
from .pipeline import create_pipe

from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV, cross_validate, StratifiedKFold


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
@click.option("--select-model", default="tree", type=click.Choice(["tree", "logreg"], case_sensitive=False), 
    show_default=True)
@click.option("--use-scaler", default=True, type=bool, show_default=True)
@click.option("--logreg-max-iter", default=50, type=float, show_default=True)
@click.option("--logreg-penalty", default="l2", type=click.Choice(["none", "l2"]), show_default=True)
@click.option("--logreg-c", default=1, type=click.FloatRange(min=0, min_open=True), show_default=True)
@click.option("--tree-crit", default="gini", type=click.Choice(["gini", "entropy"], case_sensitive=False), 
    show_default=True)
@click.option("--tree-max-depth", default=0, type=int, show_default=True)
@click.option("--tree-min-samples-leaf", default=1, type=int, show_default=True)
def train(
    dataset_path, save_model_path, test_split_ratio,
    random_state, select_model, use_scaler, 
    logreg_max_iter, logreg_penalty,logreg_c, tree_crit,
    tree_max_depth,tree_min_samples_leaf):
    X_tr, X_val, y_tr, y_val = get_data(dataset_path, test_split_ratio, random_state)
    with mlflow.start_run():
        pipe = create_pipe(select_model, random_state, use_scaler, logreg_penalty,
            logreg_max_iter, logreg_c, tree_crit, tree_max_depth, tree_min_samples_leaf)
        scorings = ("accuracy", "f1_macro", "roc_auc_ovr")
        tracking_params = {"model":select_model, "random_state":random_state,
            "use_scaler":use_scaler, "logreg_penalty":logreg_penalty,
            "logreg_max_iter":logreg_max_iter, "logreg_c":logreg_c, "tree_crit":tree_crit,
            "tree_max_depth":tree_max_depth, "tree_min_samples_leaf":tree_min_samples_leaf}
        mlflow.log_params(tracking_params)
        # outer_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=random_state)
        inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
        # clf = RandomizedSearchCV(pipe, param_grid, cv=inner_cv, n_iter=100, n_jobs=-1)
        # nested_cv = cross_validate(pipe, X_tr, y_tr, cv=outer_cv, n_jobs=-1, scoring=scorings)
        cv = cross_validate(pipe, X_tr, y_tr, cv=inner_cv, n_jobs=-1, scoring=scorings)
        # click.echo(f"nested scores {metric}={nested_score.mean()}")
        mlflow.log_metrics({
            "accuracy":cv["test_accuracy"].mean(), 
            "f1_score":cv["test_f1_macro"].mean(),
            "roc_auc_ovr":cv["test_roc_auc_ovr"].mean()})
        print({
            "accuracy":cv["test_accuracy"].mean(), 
            "f1_score":cv["test_f1_macro"].mean(),
            "roc_auc_ovr":cv["test_roc_auc_ovr"].mean()})
        # dump(pipe, save_model_path)
        # click.echo(f"model saved to {save_model_path}")
