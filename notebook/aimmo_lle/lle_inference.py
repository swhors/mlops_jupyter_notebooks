import argparse 
import glob
from datetime import datetime

# from gtgen.mlc.timm.utils import get_outdir
from gtgen.util import logger ### ! 수정

from gtgen.lle.inference import GtGenlleInference ### ? 
from gtgen.lle.config import Config ### ! 수정
import json
import os
from PIL import Image,ImageDraw,ImageFont
import datetime
import time
import numpy as np


def get_files_recursively(folder_path):
    file_list = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.split('.')[-1] in ['jpg','png','jpeg']:
                file_list.append(os.path.join(root, file))
    return file_list

def inference(args):
    start = time.time()
    args = parse_args()
    # dir_path = args.in_dir # ! /마지막에 들어가야함
    dir_path = '/home/datalab/clid/git/lle-torch/sample_img_raw'
    args.pretrained_path = '/ext/Dataset/lle/lle_model.pth'
    args.gpu = True
    img_list = get_files_recursively(dir_path) # 파일 리스트 

    if args.gpu:
        import torch
        device_id = torch.cuda.current_device()
    else:
        device_id = -1
    lle_inf = GtGenlleInference(devices=device_id,   #### ! 수정
                              save_log=args.save_log,
                              print_log=args.print_log,
                              pretrained_path=args.pretrained_path,
                              raw_image_inf=args.raw_image_inf
                            #   used_models = used_models, 
                            #   lle_config = args.lle_config
    )

    
    if not lle_inf.load_model(): 
        return

    args.save_dir = '/home/datalab/clid/git/lle-torch/result'
    for img_path in img_list:
        s = time.time()
        enhance_image_numpy = lle_inf.inference(img_path)
        # print("numpy 변환 : ", time.time() - s)
        # json_result.append(dict_res)
        # # ! -- 여기에 2장 비교하는 사진 저장하는 파일이 있으면 됨
        # file_name = img_path.split('/')[-1]
        # lle_inf.get_save_img(enhance_image_numpy, f'{args.save_dir}{file_name}')
        
        relative_path = os.path.relpath(img_path, dir_path)  # 상대 경로 구하기
        save_path = os.path.join(args.save_dir, relative_path)  # 저장 경로 설정
        os.makedirs(os.path.dirname(save_path), exist_ok=True)  # 저장 경로 생성
        lle_inf.get_save_img(enhance_image_numpy, save_path)
        
        
        # lle_inf.get_save_twin_img(np.array(Image.open(img_path)),enhance_image_numpy, f'{args.save_dir}{file_name}')
        # print("저장 까지 : ", time.time() - s)
        
    print(time.time() - start)
    

def parse_args():
    parser = argparse.ArgumentParser(
        description='Vision Analyze inference')
    parser.add_argument('--cmd-args', default=None)
    parser.add_argument('--gpu', default=False, action='store_true')
    parser.add_argument('--raw-image-inf', default=False, action='store_true')
    parser.add_argument('--in-dir', type=str,
                    help='input dir path including image files')
    parser.add_argument('--save-dir', type=str, default='/home/datalab/clid/lle-torch/not/',
                        help='the dir to save logs and models')
    parser.add_argument('--pretrained-path', help='path to pretrained model weights') 
    
    # save log
    parser.add_argument('--save-log', default=False, action='store_true',
                        help="if check, save log")
    parser.add_argument('--print-log', default=False, action='store_true',
                        help="if check, print log")
    args = parser.parse_args()
    if args.cmd_args is not None:
        with open(args.cmd_args) as f: 
            cmd_args = json.load(f)
            args = parser.parse_args(cmd_args['args'])
    return args



if __name__ == '__main__':
    args = parse_args()
    inference(args)