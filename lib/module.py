import importlib.util
import os
import sys
import platform

from importlib.util import spec_from_file_location, module_from_spec
from lib.path_utils import get_home_path_and_path_splitter


def load_function(lib_path_filename, function_name):
    # load_function
    """
    load python function from file
    :param lib_path_filename: str
    :param function_name: str
    :return: function
    """
    spec = spec_from_file_location('lib_name', lib_path_filename)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    func = getattr(module, function_name)
    return func


def _load_class_module(module_path_name):
    # load_class_module
    """
    load python class from file
    :param module_path_name: str
    :return: class
    """
    print(f"module_path_name = {module_path_name}")
    spec = spec_from_file_location('classname', module_path_name)
    print(spec, type(spec))
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    cls = getattr(module, module.classname)
    return cls


def load_dataloader(loader_name):
    # load_plugin_module
    """
    load python class from file
    :param loader_name: str
    :return: class
    """

    home, path_splitter = get_home_path_and_path_splitter(use_cur_parent=True)
    print(f"home = {home}, path_splitter = {path_splitter}")

    print("load_class_module in debug mode")
    plugin_path = f'lib{path_splitter}data_loader{path_splitter}{loader_name}.py'

    if home.endswith(path_splitter):
        module_path_name = home + plugin_path
    else:
        module_path_name = home + path_splitter + plugin_path
    return _load_class_module(module_path_name=module_path_name)


def load_plugin_module(class_name):
    # load_plugin_module
    """
    load python class from file
    :param class_name: str
    :return: class
    """

    home, path_splitter = get_home_path_and_path_splitter(use_cur_parent=True)
    print(f"home = {home}, path_splitter = {path_splitter}")

    print("load_class_module in debug mode")
    plugin_path = f'lib{path_splitter}plugins{path_splitter}{class_name}.py'

    if home.endswith(path_splitter):
        module_path_name = home + plugin_path
    else:
        module_path_name = home + path_splitter + plugin_path
    return _load_class_module(module_path_name=module_path_name)


def load_all_dynamic_module(run_mode):
    # load_all_dynamic_module
    if run_mode == "debug":
        print("load_all_dynamic_module in debug mode")
        use_cur_parent = True
    else:
        print("load_all_dynamic_module in prod mode")
        use_cur_parent = False
    home, path_splitter = get_home_path_and_path_splitter(use_cur_parent=True)
    if run_mode == "debug":
        plugin_dir = f'{home}{path_splitter}lib{path_splitter}plugins'
    else:
        plugin_dir = f'{home}{path_splitter}work{path_splitter}lib{path_splitter}plugins'
    file_list = os.listdir(plugin_dir)
    class_cols = {}
    for file in file_list:
        name_ext = file.split(".")
        if len(name_ext) == 2 and name_ext[1] == "py":
            cls = load_class_module(class_name=name_ext[0], run_mode=run_mode)
            class_cols[name_ext[0]] = cls
    return class_cols
