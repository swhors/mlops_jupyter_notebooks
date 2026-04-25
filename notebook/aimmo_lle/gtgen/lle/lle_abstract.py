import os

# from gtgen.utils import class_to_dict, logger, misc
from ..util import logger
from gtgen.lle.config import Config 
from gtgen.lle.version import __version__ as version
import json


class GtGenlleAbstract:
    def __init__(self,
                 devices: str,
                 batch_size : int,
                 num_workers : int,
                 train_config : str = '',
                #  work_dir: str = '',
                 model_name: str = 'lle'):

 
        self.cfg = Config(model_name = model_name).get_configs()()

        self.devices = devices

        # self.cfg.lleConfig.work_dir =work_dir
        
        if train_config != '':
            with open(train_config, 'r') as f: # ! 상대경로로 수정
                config = json.load(f)
            self.cfg.update(config)
        self.cfg.update_with_values(device='cpu' if devices == '-1' else 'cuda',
                                    devices=devices,
                                    batch_size=batch_size,
                                    workers = num_workers,
                                    # work_dir=work_dir,
                                    meta_name=model_name) # ! train에서는 meta가 없으니까
    
    def create_model(self,
                     args):
        
        # from gtgen.mlc.cls.lib.models.query2label import create_model as create_model_
        
    
        # model = create_model_(self.cfg) # 모델 생성
        
        # return model
        return