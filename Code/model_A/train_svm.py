# Code/model_A/train_svm.py
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

def train_model(X_train, y_train, model_type="svm"):
    if model_type == "svm":
        model = SVC(kernel="rbf", C=1.0, gamma="scale")
    elif model_type == "logistic":
        model = LogisticRegression(max_iter=1000)
    else:
        raise ValueError("Unknown model type")

    model.fit(X_train, y_train)
    return model