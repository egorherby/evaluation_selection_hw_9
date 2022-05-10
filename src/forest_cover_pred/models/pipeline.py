from sklearn import model_selection
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler, QuantileTransformer


def create_pipe(
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
):
    # param_grid = {
    #     "tree__criterion":["gini", "entropy"],
    #     "tree__splitter":["best", "random"],
    #     "tree__max_depth":[None, 3,5,7,10,15,20,30,50],
    #     "tree__min_samples_leaf":[1, 3, 5, 7, 10, 20, 25, 30],
    #     "tree__max_features":["sqrt","log2",None]
    # }
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
        return make_pipeline(
            feature_eng[feature_eng_type], scaler[use_scaler], models[select_model]
        )
    else:
        return make_pipeline(
            scaler[use_scaler], feature_eng[feature_eng_type], models[select_model]
        )
