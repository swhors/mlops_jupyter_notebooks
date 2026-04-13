from ..gtgen_abstract import *
from .lle_abstract import *

from gtgen.util import misc, logger
# from .timm.utils import init_distributed_device # ! 수정 필요
# from ..lib import dist_parallel
from multiprocessing import shared_memory
# import torch.backends.cudnn as cudnn
import datetime
from gtgen.lle.config import Config 
# import gtgen.lle.cls.impl import model


class GtGenlleEvaluation(GtGenlleAbstract, GtGenModelEvaluationAbstract):
    def __init__(self,
                 devices: int,
                 batch_size : int,
                 num_workers: int,
                 pretrained_path: str,
                 raw_image_inf: bool = False
                ):
        
        # self.data_config = None
        # self.inf_transform = None
        # self.attn_vis = False

        super().__init__(devices=devices,
                         num_workers=num_workers,
                        #  work_dir=work_dir,
                         batch_size=batch_size)
        
        self.cfg.lleConfig.devices = devices # ?
        self.cfg.lleConfig.workers = num_workers # ?
        self.cfg.lleConfig.batch_size = batch_size # ?
        self.cfg.lleConfig.pretrained_path = pretrained_path
        if raw_image_inf:
            self.cfg.lleConfig.scale_factor = 1
        else:
            self.cfg.lleConfig.scale_factor = 12
        # if 'WORLD_SIZE' in os.environ: # ! 
        #     assert self.cfg.DistributionConfig.world_size > 0, 'please set --world-size and --rank in the command line'
        #     local_world_size = int(os.environ['WORLD_SIZE'])
        #     self.cfg.DistributionConfig.world_size = self.cfg.DistributionConfig.world_size * local_world_size
        #     self.cfg.DistributionConfig.rank = self.cfg.DistributionConfig.rank * local_world_size + self.cfg.DistributionConfig.local_rank
        #     print('world size: {}, world rank: {}, local rank: {}'.format(self.cfg.DistributionConfig.world_size, self.cfg.DistributionConfig.rank, self.cfg.DistributionConfig.local_rank))
        #     print('os.environ:', os.environ)
        # elif devices == '-1':
        #     # only cpu
        #     self.cfg.DistributionConfig.world_size = -1
        #     self.cfg.DistributionConfig.rank = -1
        #     self.cfg.DistributionConfig.local_rank = -1
        # else:
        #     self.cfg.DistributionConfig.world_size = 1
        #     self.cfg.DistributionConfig.rank = 0
        #     self.cfg.DistributionConfig.local_rank = 0
        
        
        
        # ngpus_per_node = len(self.devices.split(','))
        # self.cfg.lleConfig.distributed = ngpus_per_node > 1
        # os.environ['CUDA_VISIBLE_DEVICES'] = self.devices
        # self.cfg.lleConfig.devices = self.devices 
        # self.cfg.lleConfig.world_size = ngpus_per_node
        
        
        # if self.__class__.__name__ != 'GtGenlleInference':    
        #     self.cfg.LossConfig.workers = num_workers
        #     self.devices = devices
        #     work_dir = self.cfg.DatasetConfig.work_dir
        #     work_dir = work_dir if work_dir else './work' + (f'/evaluation-{datetime.datetime.now()}').replace(' ', '-')
        #     self.cfg.ModelConfig.work_dir = self.cfg.DatasetConfig.work_dir =  work_dir
        #     os.makedirs(work_dir, exist_ok=True)
        #     self.logger = logger.setup_logger("Eval", folder=work_dir, filename="evaluation")
        #     self.logger.info(f"version={version}")
        #     misc.init_torch(train_phase = False)
        #     self.logger.info('GtGenlleEvaluation init complete.')
        # else:
            # work_dir = self.cfg.DatasetConfig.work_dir
            # work_dir = work_dir if work_dir else './work' + (f'/inference-{datetime.datetime.now()}').replace(' ', '-')
            # self.cfg.ModelConfig.work_dir = self.cfg.DatasetConfig.work_dir =  work_dir
            # if save_log:# ! inference 로거 생성
            #     os.makedirs(work_dir, exist_ok=True)
            #     self.logger = logger.setup_logger("Inf", folder=work_dir, filename="inference") 
            #     self.logger.info(f"version={version}")
            #     self.logger.info("Start Inference\n")
            # else:
            #     self.logger = None
             
            
            
        # self.model = None
        
    def load_model(self, 
                #    pretrained_path : str,
                    # scale_factor : int = 12,
                    ) -> bool:
        
        from .cls.impl.lle_model import load_model_ ### ! va_model에 있는 것인데, 일단 model이랑, state_dict를 반환함
        
        self.model = load_model_(self, self.cfg.lleConfig.scale_factor)

        if self.model:
            return True
        # if self.models:
        #     # self.model, self.inf_transform = get_infmodel(self.cfg, self.logger, self.model)
        #     self.model = True 
        #     return True
        

        self.logger.error(f"state of loaded model is invalid")
        return False
        
    def has_model(self) -> bool:
        return self.model is not None

    def _assert_no_model(self):
        assert self.model is not None, "no loaded model"

    def get_model_config(self) -> dict:
        return self.cfg.__dict__

    # for MLOps. ex) ("vehicle", "pedestrian")
    def get_model_classes(self) -> tuple:
        self._assert_no_model()
        return self.cfg.DatasetConfig.class_list
    
    def evaluation(self,
                    dataset_config: str,
                    save_path: str = ''
    ) -> dict:
        self._assert_no_model()
        
        
        from .cls.impl.validate import mlc_val
        from .cls.lib.dataset.get_dataset import create_dataset as create_dataset_
        from attrdict import AttrDict
        
        # if self.logger:
        self.logger.info("GtGenMlcEvaluation.evaluation start")
        
        with open(dataset_config, 'r') as f:
            # dataset config
            config = json.load(f)
        self.cfg.update(config)
        # if self.logger:
        self.logger.info(f"GtGenMlcEvaluation.create_dataset")
        eval_dataset = create_dataset_(self.cfg,
                                       is_training=False,
                                       img_path = self.cfg.DatasetConfig.val_dir,
                                       class_list = self.cfg.DatasetConfig.class_list,
                                        meta_name = self.cfg.DatasetConfig.meta_name,
                                        json_path = self.cfg.DatasetConfig.val_json
                                       )
        
        args = AttrDict(cfg=self.cfg,
                        model=self.model,
                        eval_dataset=eval_dataset,
                        data_config=self.data_config,
                        save_path=save_path) 
        
        try:
            self.logger.info(f'Evaluation with process on ({self.devices}) device.')

            ret = mlc_val(args, self.logger)
            self.logger.info(f"GtGenMlcEvaluation.evaluation.finish. {ret}")
        except Exception as e:
            logger.write_exception_log(
                self.logger, e, "GtGenMlcEvaluation failed to evaluation"
            )
            ret = None

        return ret