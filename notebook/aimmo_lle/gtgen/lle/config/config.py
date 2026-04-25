import os


class Config:
    def __init__(self, model_name: str = 'lle') -> None:

        if model_name == 'lle':
            from .lle.lle import Configs
        # elif model_name == 'coco':
        #     from .Swin.coco import Configs
        # elif model_name == 'swin_base':
        #     from .Swin.swin_base import Configs
        else:
            from .lle.dummy import Configs
            # raise ValueError('Model type that does not exist')

        self._configs = Configs

    def get_configs(self):
        return self._configs
