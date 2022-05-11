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
@click.option(
    "--select-model",
    default="tree",
    type=click.Choice(["tree", "knn"], case_sensitive=False),
    show_default=True,
)
@click.option(
    "--use-scaler",
    default="standard",
    type=click.Choice(["none", "standard", "quantile"]),
    show_default=True,
)
@click.option(
    "--feature-eng-type",
    default="none",
    type=click.Choice(["none", "var", "pca"]),
    show_default=True,
)
@click.option("--var-threshold", default=1e-3, show_default=True)
@click.option("--pca-n-features", default=40, show_default=True)
@click.option("--knn-neighbors", default=10, show_default=True)
@click.option(
    "--knn-weights",
    default="uniform",
    type=click.Choice(["uniform", "distance"]),
    show_default=True,
)
@click.option(
    "--tree-crit",
    default="gini",
    type=click.Choice(["gini", "entropy"], case_sensitive=False),
    show_default=True,
)
@click.option("--tree-max-depth", default=0, type=int, show_default=True)
@click.option("--tree-min-samples-leaf", default=1, type=int, show_default=True)
@click.option("--find-best-params", default=False, type=bool, show_default=True)
def train(
    dataset_path,
    save_model_path,
    test_split_ratio,
    select_model,
    random_state,
    use_scaler,
    feature_eng_type,
    pca_n_features,
    var_threshold,
    knn_neighbors,
    knn_weights,
    tree_crit,
    tree_max_depth,
    tree_min_samples_leaf,
    find_best_params,
):
    X_tr, X_val, y_tr, y_val = get_data(dataset_path, test_split_ratio, random_state)
    
    scorings = ("accuracy", "f1_macro", "roc_auc_ovr")

    with mlflow.start_run():
        if find_best_params:
            click.echo("finding best parameters using randomized search")
            click.echo(f"using `{select_model}` model")
            if select_model == "knn":
                pipe = create_pipe(select_model, random_state, 'standard', 'pca')
                param_grid = {
                    "knn__n_neighbors": list(range(1, 30)),
                    "knn__weights":["uniform", "distance"],
                    "feat_eng__n_components":list(range(45, 52))
                }
            else:
                pipe = create_pipe(select_model, random_state)
                param_grid = {
                    "tree__criterion":["gini", "entropy"],
                    "tree__max_depth":[None, 3,5,7,10,15,20,30,50],
                    "tree__min_samples_leaf":[1, 3, 5, 7, 10, 20, 25, 30],
                    "tree__max_features":["sqrt","log2",None]
                }
            outer_loop = StratifiedKFold(n_splits=3, shuffle=True, random_state=random_state)
            inner_loop = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
            clf = RandomizedSearchCV(pipe, param_grid, cv=inner_loop, n_iter=10, n_jobs=-1, scoring='f1_macro')
            nested_cv = cross_validate(pipe, X_tr, y_tr, cv=outer_loop, n_jobs=-1, scoring=scorings)
            click.echo('nested cross validation finished')
            cv_average_metrics = {
                "accuracy": nested_cv["test_accuracy"].mean(),
                "f1_score": nested_cv["test_f1_macro"].mean(),
                "roc_auc_ovr": nested_cv["test_roc_auc_ovr"].mean(),
            }
            mlflow.log_metrics(cv_average_metrics)
            click.echo(f"unbiased averaged nested cv metrics{cv_average_metrics}")
            click.echo("retraining on whole training_set to obtain best parameters....")
            clf.fit(X_tr, y_tr)
            f1_best = clf.score(X_val, y_val)
            mlflow.log_params(clf.best_params_)
            mlflow.log_metric("f1 after retrain", f1_best)
            click.echo(f"best parameters{clf.best_params_}")
            click.echo(f"best estimator score on holdout set={f1_best}")
        else:
            pipe = create_pipe(
                select_model,
                random_state,
                use_scaler,
                feature_eng_type,
                pca_n_features,
                var_threshold,
                knn_neighbors,
                knn_weights,
                tree_crit,
                tree_max_depth,
                tree_min_samples_leaf,
            )
            tracking_params = {
                "select_model": select_model,
                "random_state": random_state,
                "use_scaler": use_scaler,
                "feature_eng_type": feature_eng_type,
                "pca_n_features": pca_n_features,
                "var_threshold": var_threshold,
                "knn_neighbors": knn_neighbors,
                "knn_weights": knn_weights,
                "tree_crit": tree_crit,
                "tree_max_depth": tree_max_depth,
                "tree_min_samples_leaf": tree_min_samples_leaf,
            }
            mlflow.log_params(tracking_params)
            inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
            cv = cross_validate(pipe, X_tr, y_tr, cv=inner_cv, n_jobs=-1, scoring=scorings)
            cv_average_metrics = {
                "accuracy": cv["test_accuracy"].mean(),
                "f1_score": cv["test_f1_macro"].mean(),
                "roc_auc_ovr": cv["test_roc_auc_ovr"].mean(),
            }
            click.echo(f"cv averaged metrics {cv_average_metrics}")
            mlflow.log_metrics(cv_average_metrics)
        # dump(pipe, save_model_path)
        # click.echo(f"model saved to {save_model_path}")
