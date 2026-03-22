from lib.plugin_base import PluginBase
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler


classname = "SVC"


class SVC(PluginBase):
    def __init__(self):
        pass

    def _pre_proc(self, X_train, X_test, y_train, y_test):
        pass

    def _post_proc(self, trained_model):
        pass

    def _str_(self):
        pass
    
    def _load_data(self, data_uri: str, params: dict = {}):
        pass
    
    def _model_save(self, model_name, trained_model, store_uri):
        pass
    
    def _get_metric(self, model_name, trained_model, store_uri):
        pass

    def _train(self, X_train, X_test=None, y_train=None, y_test=None, parameters={}):
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        model = SVC(**parameters)
        model.fit(X_train_scaled, y_train)
        return model