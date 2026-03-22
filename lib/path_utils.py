import platform
import os
import sys
from pathlib import Path
from lib.types import RunMode


def get_home_by_path(upper_step=2):
    current_path = os.path.abspath(__file__)
    print(f"#00 current_path = {current_path}")
    parent_dir = os.path.dirname(current_path)
    print(f"#01 parent_dir = {parent_dir}")
    # 두 단계 위 상위 디렉토리]
    for step in range(upper_step-1):
        grandparent_dir = os.path.dirname(parent_dir)
        if grandparent_dir is not None:
            parent_dir = grandparent_dir
        else:
            break
    return grandparent_dir


def get_home_path_and_path_splitter(use_cur_parent = False, upper_step=2):
    path_splitter = os.sep

    if use_cur_parent:
        home = get_home_by_path(upper_step)
    else:
        if platform.system() == "Windows":
            home = os.environ.get("HOMEPATH")
            # path_splitter = "\\"
        else:
            home = os.environ.get("HOME")
            # path_splitter = "/"
        if home is None:
            home = str(Path.home())
    print(f'home = {home}, path_splitter = {path_splitter}')
    return home, path_splitter
