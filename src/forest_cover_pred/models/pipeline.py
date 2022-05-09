from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier


def create_pipe():
    p = Pipeline([
        ("tree", DecisionTreeClassifier())
    ]
    )
    return p
