import json
import enum

from .logger import get_stdout_logger

class ClassToDict:
    def __init__(self, package_name, cls_inst, log=get_stdout_logger()) -> None:
        self.package_name = package_name
        self.cls_inst = cls_inst
        self.log = log

    def from_path_or_dict(self, path_or_dict):
        if isinstance(path_or_dict, str):
            if self.from_jsonfile(path_or_dict) is not self.cls_inst:    # impossible
                self.log.info(f"failed to load:{path_or_dict}")
                return False
        elif isinstance(path_or_dict, dict):
            if self.from_dict(path_or_dict) is not self.cls_inst:
                self.log.info(f"failed to load:{path_or_dict}")
                return False
        else:
            return False
        return True
                
    def from_dict(self, d):
        return self.__from_dict(d, self.cls_inst)
    
    def from_jsonfile(self, json_path):
        with open(json_path) as f:
            d = json.load(f)
            return self.from_dict(d)
    
    def to_dict(self):
        return self.__to_dict(self.cls_inst)
    
    def __from_dict(self, d, target):
        for k, v in d.items():
            if not hasattr(target, k):
                continue
            
            if v is not None:
                target_v = getattr(target, k)
                if target_v is not None:
                    if isinstance(v, dict):
                        if not isinstance(target_v, dict):  # if value type is dict and target value type is not dict, it might be class.
                            v = self.__from_dict(v, getattr(target, k))
                    elif isinstance(v, str):
                        if isinstance(target_v, enum.Enum):
                            v = target_v.__class__[v]   # string to value of target enum class

            setattr(target, k, v)
        return target
    
    def __to_dict(self, source):
        ret = dict()
        for k in source.__dict__:
            v = getattr(source, k)
            
            if not isinstance(v, builtin_types):
                if isinstance(v, enum.Enum):
                    v = v.name
                elif is_pkg_class(v):
                    v = self.__to_dict(v)
                else:
                    continue

            ret[k] = v
        return ret
    
    def __str__(self) -> str:
        cfg_dict = self.to_dict()
        return json.dumps(cfg_dict, sort_keys=False, indent=4)
        
def is_pkg_class(attr_v):
    if hasattr(attr_v, "__module__"):
        if "config" in attr_v.__module__:
            return True
        else:
            print("check this!!")
    elif hasattr(attr_v.__class__, "__base__"):
        if isinstance(attr_v.__class__.__base__, object):
            b = isinstance(attr_v, object)
            
    return False

builtin_types = (int, float, str, list, tuple, dict, set)

def fill_config_from_args(args, target):
    import enum
    for name in target.__dict__:
        if name.startswith('_'):
            continue

        target_v = getattr(target, name)
        if is_pkg_class(target_v):
            fill_config_from_args(args, target_v)
            continue
        
        if not hasattr(args, name):
            continue
        
        arg_v = getattr(args, name)
        if target_v is None: # unable to check type
            pass
        else:
            if isinstance(target_v, enum.Enum):
                arg_v = target_v.__class__[arg_v]
            elif isinstance(target_v, (list, tuple)):
                arg_v = [item for item in arg_v.split(',')]
                if len(target_v) > 0:
                    if isinstance(target_v[0], (int, float)):
                        arg_v = [target_v[0].__class__(x) for x in arg_v]
            elif isinstance(target_v, (int, float, str)):
                arg_v = target_v.__class__(arg_v)
            else:   # including dict
                a = 0  
            
        setattr(target, name, arg_v)
