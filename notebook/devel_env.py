import os
import sys

def init_env():
    """
    # 개발 환경 초기화
    # 한 번만 호출하여 주십시요.

    """
    # 1.1 set pythonpath to import from parent directory
    parent_dir = os.path.dirname(os.getcwd())
    if os.environ.get("PYTHONPATH") is None:
        os.environ["PYTHONPATH"] = ""
    if parent_dir not in sys.path:
        sys.path.append(parent_dir)
        os.environ["PYTHONPATH"] = os.environ["PYTHONPATH"] + ";" + parent_dir
    python_def_lib_path = os.path.join(parent_dir, "venv\\Lib\\site-packages")
    if python_def_lib_path not in sys.path:
        sys.path.append(python_def_lib_path)
        os.environ["PYTHONPATH"] = os.environ["PYTHONPATH"] + ";" + python_def_lib_path
    os.environ["CLEARML_WEB_HOST"] = "http://172.16.8.168:8080/"
    os.environ["CLEARML_API_HOST"] = "http://172.16.8.168:8008/"
    os.environ["CLEARML_FILES_HOST"] = "http://172.16.8.168:8081"
    os.environ["CLEARML_API_ACCESS_KEY"] = "SIJ8V8YP9PL25YEAVAWGP9NV11TMOQ"
    os.environ["CLEARML_API_SECRET_KEY"] = "6x_MEHvgrUv9-TFFDrE9BE7vb3JYOXDWq4kQtqbd58nQ5pqHEGMt6qQXxF7_HCyzq7E"