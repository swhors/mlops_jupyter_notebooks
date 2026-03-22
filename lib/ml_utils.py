"""
ml_param_util

입력 파라미터의 유틸리티를 정의합니다.
"""

from enum import Enum


def str_to_ml_parameters(param_str: str):
    """
    str_to_ml_parameters
      스트링 형태의 파라미터를 dict 타입으로 변환하여 반환
    
    input 
        param_str: str , string 형태의 파라미터 값
                   (ex: n_estimators=100,max_depth=5)
    return
        parameters: dict
    """
    parameters = {}
    parameter_list = param_str.split(",")
    
    for parameter in parameter_list:
        key_value = parameter.split("=")
        if len(key_value) == 2:
            key, value = key_value
            parameters[key] = int(value) if value.isdigit() else value
    
    return parameters


def load_model_class(class_uri: str):
    class_info = class_uri.split("://")

    X_train = None
    X_test = None
    y_train = None
    y_test = None

    if len(class_info) == 2:
        pass
    else:
        print(f"Error: illegal uri (uri = {class_uri})")

    return X_train, X_test, y_train, y_test
    