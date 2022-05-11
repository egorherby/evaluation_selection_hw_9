from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler, QuantileTransformer


def create_pipe(
    select_model,
    random_state=42,
    use_scaler="none",
    feature_eng_type="none",
    pca_n_features=1,
    var_threshold=1e-3,
    knn_neighbors=10,
    knn_weights="uniform",
    tree_crit="gini",
    tree_max_depth=0,
    tree_min_samples_leaf=1,
):
    if tree_max_depth == 0:
        tree_max_depth = None
    scaler = {
        "none": "passthrough",
        "quantile": QuantileTransformer(output_distribution="normal"),
        "standard": StandardScaler(),
    }
    feature_eng = {
        "none": "passthrough",
        "pca": PCA(pca_n_features),
        "var": VarianceThreshold(threshold=var_threshold),
    }
    models = {
        "knn": KNeighborsClassifier(
            n_neighbors=knn_neighbors, weights=knn_weights, n_jobs=-1
        ),
        "tree": DecisionTreeClassifier(
            random_state=random_state,
            criterion=tree_crit,
            max_depth=tree_max_depth,
            min_samples_leaf=tree_min_samples_leaf,
        ),
    }
    if feature_eng_type == "var":
        return Pipeline(
            [
                ("feat_eng", feature_eng[feature_eng_type]),
                ("scaler", scaler[use_scaler]),
                (select_model, models[select_model]),
            ]
        )
    else:
        return Pipeline(
            [
                ("scaler", scaler[use_scaler]),
                ("feat_eng", feature_eng[feature_eng_type]),
                (select_model, models[select_model]),
            ]
        )
