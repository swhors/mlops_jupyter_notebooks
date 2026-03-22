import os
import sys
import platform
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import joblib
from clearml import Task

def _train(X_train, y_train, parameters):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    svc = SVC(**parameters)
    svc.fit(X_train_scaled, y_train)
    return svc