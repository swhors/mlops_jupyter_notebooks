# ! 결과 나오는 

import argparse
import os, sys
import random
import datetime
import time
from typing import List
import json
import numpy as np
# import pandas as pd
import os

# import torch
# import torch.nn as nn
# import torch.nn.parallel
# import torch.backends.cudnn as cudnn
# import torch.distributed as dist
# import torch.optim
# import torch.utils.data
# import torch.utils.data.distributed

# from IPython.display import display, Image
from PIL import Image, ImageOps
from tqdm import tqdm
# import torchvision.transforms as transforms
import torch
from gtgen.util.timer import Timer 


def lle_inference(cfg,  
                  # logger,
                  model,
                  input,
                  rate,
                  scale_factor
                ) : 
    
    timer = Timer()
    timer.tic()

    data_lowlight = ImageOps.exif_transpose(input)
    data_lowlight = (np.asarray(data_lowlight)/255.0)

    data_lowlight = torch.from_numpy(data_lowlight).float()
    h=(data_lowlight.shape[0]//scale_factor)*scale_factor
    w=(data_lowlight.shape[1]//scale_factor)*scale_factor
    data_lowlight = data_lowlight[0:h,0:w,:]
    data_lowlight = data_lowlight.permute(2,0,1)
    if cfg.lleConfig.devices == -1:
      data_lowlight = data_lowlight.unsqueeze(0)
    else:
      data_lowlight = data_lowlight.cuda().unsqueeze(0) # ! 원본
    enhanced_image, params_maps = model(data_lowlight, rate)

    duration = timer.toc()
    fps = 1 / duration 
    speed = duration / 1
    print(f"rate : {rate} / {1} bath inference :\tfps:{fps}, speed:{speed}")

    return enhanced_image
