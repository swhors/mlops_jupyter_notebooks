from lib.plugin_base import PluginBase

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


classname = "RandomForest"


class RandomForest(PluginBase):
    def __init__(self):
        pass

    def _pre_proc(self, X_train, X_test, y_train, y_test):
        return X_train, X_test, y_train, y_test

    def _post_proc(self, trained_model):
        super()._post_proc(trained_model=trained_model)

    def _str_(self):
        pass
    
    def _load_data(self, data_uri: str, params: dict = {}):
        return super()._load_data(data_uri=data_uri, params=params)
    
    def _model_save(self, model_name, trained_model, store_uri):
        pass
    
    def _metric_save(self, model_name, trained_model, store_uri):
        pass

    def _train(self, X_train, X_test=None, y_train=None, y_test=None, parameters={}):
        # train random forest model
        print('RandomForest._train.X_train=', len(X_train))
        print('RandomForest._train.y_train=', len(y_train))
        model = RandomForestClassifier(**parameters)
        model.fit(X_train, y_train)
        return model