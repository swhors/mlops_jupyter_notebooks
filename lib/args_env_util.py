"""
args_env_util

프로그램의 입력 인자와 env 값을 가져오거나 설정하는 util 기능을 정의합니다.
"""

import argparse
import os


def get_arguments():
    parser = argparse.ArgumentParser(description="arguments for sub_process")
    
    # 인자 추가
    parser.add_argument('--class_name', type=str, help='플러그인 클래스 이름', default="Sample")
    parser.add_argument('--train_params', type=str, help='파라미터', default="")
    parser.add_argument('--data_params', type=str, help='파라미터', default="")    
    parser.add_argument('--data_uri', type=str, help='데이터 URI', default="")
    parser.add_argument('--result_uri', type=str, help='result URI', default="")
    parser.add_argument('--project_name', type=str, help='project name', default="standard_project")
    parser.add_argument('--use_clearml', type=str, help='use clearml', default="Yes")

    args = parser.parse_args()
    return args


def get_envs():
    project_name = os.environ.get("PROJECT_NAME", "")
    train_parameter_list = os.environ.get("TRAIN_PARAMETERS", "")
    data_parameter_list = os.environ.get("DATA_PARAMETERS", "")
    class_name = f"{os.environ.get("CLASS_NAME", "")}"
    run_mode = os.environ.get("RUN_MODE", "debug")
    data_uri = os.environ.get("DATA_URI", "")
    result_uri = os.environ.get("RESULT_URI", "")
    use_clearml = os.environ.get("USE_CLEARML", "")


    return {
        "project_name": project_name,
        "task_name": project_name + "_task",
        'train_params': train_parameter_list,
        'data_params': data_parameter_list,
        'class_name': class_name,
        'run_mode': run_mode,
        'data_uri': data_uri,
        'result_uri': result_uri,
        'use_clearml': use_clearml
    }
