import abc
import enum
import shutil

class GtGenException(Exception):
    def __init__(self, msg:str, logger=None):
        super().__init__(str)
        if logger is not None:
            logger.error(str)


class PhaseType(enum.Enum):
    train = 0
    eval = 1
    inferece = 2


class EarlyStopping:
    def __init__(self, patience=10, delta=0.01) -> None:
        self.patience = patience      # if patience <= 0, early stopping will not work
        self.delta = delta


class HyperParameters:
    def __init__(self, lr:float, max_epochs:int, early_stoppingping:EarlyStopping=None):
        self.init_lr = lr # float
        self.max_epochs = max_epochs
        self.early_stoppingping = early_stoppingping


class DefaultPrintLogger:
    def __init__(self):
        self.info = print
        self.error = print

class ProgressCallBackAbstract(metaclass=abc.ABCMeta):
    def __init__(self):
        self.stopping = False

    # must be implemented from user who uses this model
    def __call__(self, **kwargs) -> None:
        raise NotImplementedError

    # call when stopping
    def set_stop(self):
        self.stopping = True

    # use when model check whether stop or continue
    def is_stopping(self):
        return self.stopping
        
class GtGenModelLearnAbstract:
    @abc.abstractmethod
    def create_dataset(self,
                       phase:PhaseType,
                       **kwargs):
        raise NotImplementedError
        
    @abc.abstractmethod
    def train(self,
              train_dataloader,        
              valid_dataloader,
              pretrained_path,             
              resume:bool,
              progress_callback=None,
              save_path=None,
              **kwargs
              ):
        """
        best_loss=None
        best_model_path=None
        return best_loss, best_model_path
        """
        raise NotImplementedError
        
    @abc.abstractmethod
    def save_model(self, save_path) -> bool:
        raise NotImplementedError


class GtGenModelEvaluationAbstract:
    @abc.abstractmethod
    def has_model(self) -> bool:
        raise NotImplementedError

    @abc.abstractmethod
    def _assert_no_model(self):
        raise NotImplementedError
        
    @abc.abstractmethod
    def get_model_config(self) -> dict:
        raise NotImplementedError

    @abc.abstractmethod
    def get_model_classes(self) -> tuple:
        raise NotImplementedError

    # @abc.abstractmethod #! 있어야 함
    # def load_model(self, load_path) -> bool:
    #     raise NotImplementedError
    
    @abc.abstractmethod
    def evaluation(self,
                   progress_callback=None,
                   **kwargs) -> dict:
        raise NotImplementedError
    

class GtGenModelInferenceAbstract:
    @abc.abstractmethod
    def load_model(self, load_path) -> bool:
        raise NotImplementedError

    @abc.abstractmethod
    def inference(self,
                  parent_path,
                  input_path_list: list,
                  progress_callback=None,
                  ) -> list:
        raise NotImplementedError
