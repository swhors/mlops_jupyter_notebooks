import os
import sys
import platform
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib
from clearml import Task

def _train(X_train, y_train=None, parameters={}):
    # train random forest model
    model = RandomForestClassifier(**parameters)
    model.fit(X_train, y_train)
    return model
