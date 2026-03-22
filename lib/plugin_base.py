#################################
# sample code
#
from abc import ABC, abstractmethod
from lib.types import UriType
from lib.module import load_dataloader


classname="PluginBase"


class PluginBaseIf(ABC):
    @abstractmethod
    def _pre_proc(self, X_train, X_test, y_train, y_test):
        pass

    @abstractmethod
    def _post_proc(self, trained_model):
        pass

    @abstractmethod
    def _str_(self):
        pass
    
    @abstractmethod
    def _load_data(self, data_uri: str, params: dict = {}):
        pass
    
    @abstractmethod
    def _model_save(self, model_name, trained_model, store_uri):
        pass
    
    @abstractmethod
    def _metric_save(self, model_name, trained_model, store_uri):
        pass

    @abstractmethod
    def _train(self, X_train, X_test=None, y_train=None, y_test=None, parameters={}):
        pass


class PluginBase(PluginBaseIf):
    """
    sample class
    """
    def __init__(self):
        pass

    def _pre_proc(self, X_train, X_test, y_train, y_test):
        return X_train, X_test, y_train, y_test

    def _post_proc(self, trained_model):
        pass

    def _str_(self):
        pass
    
    def _load_data(self, data_uri: str, params: dict={}):
        types = UriType.get_types(data_uri)
        X_train = None
        X_test = None
        y_train = None
        y_test = None
        print(f"PluginBase._load_data.step#0")
        print(f"PluginBase._load_data.data_uri={data_uri}")
        match types:
            case UriType.URI_CLASS:
                class_file_name = data_uri.split("://")[1]
                print(f"PluginBase._load_data.step#02")
                loader = load_dataloader(class_file_name)()
                print(f"PluginBase._load_data.loader={loader}")
                if loader is not None:
                    X_train, X_test, y_train, y_test = loader.load(uri=data_uri, params=params)
        return X_train, X_test, y_train, y_test
    
    def _model_save(self, model_name, trained_model, store_uri):
        pass
    
    def _metric_save(self, model_name, trained_model, store_uri):
        pass

    def _train(self, X_train, X_test=None, y_train=None, y_test=None, parameters={}):
        pass

    def post_proc(self, trained_model):
        return self._post_proc(trained_model)
    
    def pre_proc(self, X_train, X_test, y_train, y_test):
        return self._pre_proc(X_train, X_test, y_train, y_test)
    
    def load_data(self, data_uri: str, params: dict={}):
        return self._load_data(data_uri=data_uri, params=params)
    
    def model_save(self, model_name, trained_model, store_uri):
        return self._model_save(model_name, trained_model, store_uri)
    
    def metric_save(self, model_name, trained_model, store_uri):
        return self._metric_save(model_name, trained_model, store_uri)
    
    def train(self, X_train, X_test=None, y_train=None, y_test=None, parameters={}):
        return self._train(X_train=X_train, y_train=y_train, parameters=parameters)
    
    def __str__(self):
        return self._str_()
