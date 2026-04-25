# from gtgen.utils import misc, logger, class_to_dict
from ..util import logger
# from ..utils import misc, logger, class_to_dict
from .evaluation import *

import numpy as np


from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
# 출처: https://deep-deep-deep.tistory.com/34 [딥딥딥:티스토리]
from PIL import Image
ImageFile.LOAD_TRUNCATED_IMAGES = True
# from torchvision.transforms import functional as F
# from torchvision.utils import save_image
from PIL import Image,ImageDraw,ImageFont
import time

class GtGenlleInference(GtGenlleEvaluation, GtGenModelInferenceAbstract): 
    def __init__(self,
                #  used_models: list,
                 devices: int = '0',
                 save_log: bool = False,
                 print_log: bool = False,
                 pretrained_path: str = '',
                 raw_image_inf: bool = False
                #  lle_config : str = ''
                ):
        # self.scale_factor = 12
        

        super().__init__(devices = devices, 
                         batch_size = 1,
                         num_workers = 1,
                         pretrained_path = pretrained_path,
                         raw_image_inf=raw_image_inf
                        )
                        #  work_dir = work_dir)
        


        # work_dir = self.cfg.lleConfig.work_dir
        # work_dir = work_dir if work_dir else './work' + (f'/inference-{datetime.datetime.now()}').replace(' ', '-')
        # # self.cfg.ModelConfig.work_dir = self.cfg.DatasetConfig.work_dir =  work_dir
        # self.cfg.lleConfig.work_dir = work_dir
        # if save_log:
        #     os.makedirs(work_dir, exist_ok=True)
        # self.logger = logger.setup_logger("Inf", folder=work_dir, filename="inference", save_=save_log, console_=print_log) 
        # self.logger.info(f"version={version}")
        # self.logger.info("Start Inference\n")
     # ! -- tensor -> numpy
    def tensor_to_np(
        self,
        tensor_
                     ):
        img_np = tensor_.detach().cpu().numpy() # ! -- numpy 로 바뀜
        data_uint8 = (img_np * 255).astype(np.uint8)
        tensor_color = np.transpose(data_uint8[0], (1, 2, 0)) 
        # image = Image.fromarray(tensor_color)
        return tensor_color
    
    
    # ! -- 증가 이미지 저장
    def get_save_img(
        self,
        img_arr_np,
        file_path = ''
        ):
        
        image = Image.fromarray(img_arr_np)

        if file_path:
            image.save(file_path)

        return image
    
    # ! -- 이미지 쌍으로 저장
    def get_save_twin_img(
        self,
        raw_img_np,
        lle_img_np,
        file_path = ''
    ):
        if type(raw_img_np) == np.ndarray: 
            # raw_image = F.to_pil_image(raw_img_np) 
            raw_image = Image.fromarray(raw_img_np)

        lle_image = Image.fromarray(lle_img_np)
        
        w, h  = raw_image.size
        result = Image.new("RGB",(2*w,h))
        result.paste(im=raw_image, box=(0, 0)) # ! 원본 이미지 먼저 붙이기
        result.paste(im=lle_image, box=(w,0)) # ! lle 이미지 붙이기 
        # result.save(file_path) # ! 저장 
        if file_path:
            result.save(file_path)
        
        return result
    
    # return list of result dicts
    def inference(
        self,
        input_img_path: str,
        # inf_models : list,
    ) -> np.ndarray:
        """_summary_

        Args:
            input_img_path (str): img path

        Returns:
            dict: return inference_img function
        """

        self._assert_no_model()
        
        file_name = input_img_path.split('/')[-1]
        ret = self.inference_img(Image.open(input_img_path)) 
        
        return ret
    

    # ! -- ndarray로 추론
    # If you put only the image array, the img path is not stored in the dict res
    def inference_img(
        self,
        img, 
        max_size = 1024
        # inf_models
    ) -> np.ndarray:
        
        self._assert_no_model()

        from .cls.impl import lle_inference
        
        if type(img) == np.ndarray: 
            img = Image.fromarray(img)
            # img = F.to_pil_image(img) 

        # 이미지 리사이즈 처리
        max_size = (max_size, max_size)  # 최대 크기 설정 (너비, 높이)
        if img.size[0] > max_size[0] or img.size[1] > max_size[1]:
            img.thumbnail(max_size, Image.Resampling.LANCZOS)
        
        enhance_image_tensor = lle_inference(  
            self.cfg,
            # self.logger,
            self.model,
            img,
            rate = self.cfg.lleConfig.rate,
            scale_factor= self.cfg.lleConfig.scale_factor
        )

        result_np = self.tensor_to_np(enhance_image_tensor)
        return result_np

