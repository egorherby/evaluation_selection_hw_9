from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def create_pipe(select_model, random_state, use_scaler, logreg_penalty,
        logreg_max_iter, logreg_c, tree_crit, tree_max_depth, tree_min_samples_leaf):
    # param_grid = {
    #     "tree__criterion":["gini", "entropy"],
    #     "tree__splitter":["best", "random"],
    #     "tree__max_depth":[None, 3,5,7,10,15,20,30,50],
    #     "tree__min_samples_leaf":[1, 3, 5, 7, 10, 20, 25, 30],
    #     "tree__max_features":["sqrt","log2",None]
    # }
    scaler = [("scaler", StandardScaler())]
    models = {
        "logreg":[
            ("log", LogisticRegression(random_state=random_state, penalty=logreg_penalty, C=logreg_c, max_iter=logreg_max_iter))
        ],
        "tree":[
            ("tree", DecisionTreeClassifier(random_state=random_state, criterion=tree_crit, max_depth=tree_max_depth, min_samples_leaf=tree_min_samples_leaf))
        ]
    }
    steps = models[select_model]
    if use_scaler:
        steps = scaler + steps
    return Pipeline(steps)

