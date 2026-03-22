from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


def _data_load():
    # -------------------------------
    # 4. Load dataset
    # -------------------------------
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=0.2, random_state=42
    )
    return X_train, X_test, y_train, y_test