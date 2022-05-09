from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeClassifier

def create_pipe():
    p = make_pipeline(
        DecisionTreeClassifier()
    )
    return p