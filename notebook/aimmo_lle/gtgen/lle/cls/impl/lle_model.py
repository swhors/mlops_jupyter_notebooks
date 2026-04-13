import torch
# from gtgen.mc.timm.models import create_model as timm_create_model
# from 
# from gtgen.mlc.cls.lib.models.query2label import create_model
# from gtgen.mlc.cls.lib.utils.misc import clean_state_dict
# import torch.distributed as dist
# import torch.nn.parallel
# import torch.backends.cudnn as cudnn
import os
# from ..lib.utils.logger import setup_logger
import sys
from gtgen.lle.cls.impl import model

def load_model_(self,
                scale_factor): # used_models에는 (name, path가 들어오면 됨)
    
    # print(scale_factor)
    
    if self.cfg.lleConfig.devices == -1:
        lle_model = model.lle_model(scale_factor).to('cpu')
        lle_model.load_state_dict(torch.load(self.cfg.lleConfig.pretrained_path, map_location=torch.device('cpu')))
    else:
        # lle_model = model.lle_model(scale_factor).to('cuda')
        lle_model = model.lle_model(scale_factor).cuda(self.cfg.lleConfig.devices)
        lle_model.load_state_dict(torch.load(self.cfg.lleConfig.pretrained_path))
    
    return lle_model