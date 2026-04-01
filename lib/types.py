from enum import Enum


class RunMode(Enum):
    DEBUG = 1
    PRODUCTION = 2

    @classmethod
    def get_types(cls, type_str: str):
        if type_str.startswith("debug"):
            return cls.DEBUG
        return cls.PRODUCTION


class UriType(Enum):
    URI_UNKNOWN = 0
    URI_CLASS = 1
    URI_PSQL = 2
    URI_MYSQL = 3
    URI_NFS = 4

    @classmethod
    def get_types(cls, uri_str: str):
        if uri_str.startswith("class://"):
            return cls.URI_CLASS
        if uri_str.startswith("postgres://"):
            return cls.URI_PSQL
        if uri_str.startswith("mysql://"):
            return cls.URI_MYSQL
        if uri_str.startswith("nfs://"):
            return cls.URI_NFS
        return cls.URI_UNKNOWN


def unpack_nptype(np_value):
    import numpy
    import numpy as np
    if isinstance(np_value, (numpy.float32, numpy.float64, np.float32, np.float64,
                             numpy.int32, numpy.int64, np.int32, np.int64,
                             numpy.ndarray, np.ndarray)):
        return np_value.item()
    return np_value
