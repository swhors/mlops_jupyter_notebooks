#################################
# sample code
#
from abc import ABC, abstractmethod


classname="DataLoaderBase"


class DataLoaderBase(ABC):
    """
     class
    """
    def __init__(self):
        pass

    @abstractmethod
    def _load(self, uri: str, params: dict):
        return None, None, None, None

    def load(self, uri: str, params: dict):
        return self._load(uri=uri, params=params)
