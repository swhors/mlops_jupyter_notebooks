# from gtgen.utils import logger
import collections
import json

def deep_update(src, ovr):
    for key, value in ovr.items():
        if isinstance(value, collections.abc.Mapping) and value:
            returned = deep_update(src.get(key, {}), value)
            src[key] = returned
        else:
            src[key] = ovr[key]
    return src

# ! -- config파일을 통해서 questions와 cates를 업데이트해줌
def deep_update_add(src, ovr): 
    for key, value in ovr.items():
        if isinstance(value, collections.abc.Mapping) and value:
            returned = deep_update_add(src.get(key, {}), value)
            src[key] = returned
        else:
            if key in ["questions", "cates"]:
                src[key].extend(ovr[key])
            else:
                src[key] = ovr[key]
            
    return src


def json2class(d, t):
    for k, v in d.items():
        # if k == 'checkpoint':
        #     continue
        if type(v) == dict:
            
            setattr(t, k, json2class(v, getattr(t, k)))
        else:
            setattr(t, k, v)
    return t
    
# lle parameters 
class lleConfig:
    def __init__(self) -> None:
        self.pretrained_path = ''
        self.scale_factor =  12
        self.rate = 2.0
        
        
    
# # Dataset parameters
# class DatasetConfig:
#     def __init__(self) -> None:
#         self.dataset = 'aimmo'  # dataset type
#         self.train_split = 'train'  # dataset train split
#         self.val_split = 'validation'  # dataset validation split
#         # Allow download of dataset for torch/ and tfds/ datasets that support it
#         self.dataset_download = True
#         # ! attribute class
#         # time model
#         self.meta_name = 'weather'
#         self.class_list = ('cloudy', 'sunny', 'rain')
#         # self.meta_name = 'time'
#         # self.class_list = ('day', 'night')
#         # road feature model
#         # self.meta_name = 'road_feature'
#         # self.class_list = ('r_cityroad', 'r_expressway')
#         # ! require input as args
#         self.train_dir = None
#         self.val_dir = None
#         # self.data_dir = None      # path to dataset (root dir)
#         self.train_json = None    # aimmo_labelers train json file
#         self.val_json = None      # aimmo_labelers val/eval json file
#         # self.class_map = None     # path to class to idx mapping file
#         # """
#         # {
#         #     "time_day" : 0,
#         #     "time_night" : 1
#         # }
#         # """



class Configs():
    def __init__(self) -> None:
        self.lleConfig = lleConfig()
        # self.DatasetConfig = DatasetConfig()
        # self.ModelConfig = ModelConfig()
        # self.AugConfig = AugConfig()
        # self.TransformerConfig = TransformerConfig()
        # self.DistributionConfig = DistributionConfig()
        # self.LossConfig = LossConfig()
        
    def toJSON(self):
        return json.loads(json.dumps(self,default=lambda o:o.__dict__,indent=4))
                
    def update(self, config):
        if type(config) != dict:
            config = config.toJSON()
        src_dict = self.toJSON()
        deep_update(src_dict, config)
        json2class(src_dict, self)

    # ! -- questions와 cates 업데이트 하는 메소드
    def update_add(self, config):
        if type(config) != dict:
            config = config.toJSON()
        src_dict = self.toJSON()
        deep_update_add(src_dict, config)
        json2class(src_dict, self)


    def update_with_values(self, **kwargs) -> None:
        for key in kwargs.keys():
            for config_key in self.__dict__:
                if key in self.__dict__[config_key].__dict__.keys():
                    self.__dict__[config_key].__dict__[key] = kwargs[key]
                    break