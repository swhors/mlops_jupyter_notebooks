import os

import random
import numpy as np


def init_torch(train_phase=False):
    import torch
    if torch.cuda.is_available:
        torch.set_num_threads(1)
        torch.backends.cudnn.benchmark = train_phase
        torch.backends.cuda.matmul.allow_tf32 = True


def set_random_seed(seed, deterministic=False):
    import torch
    """Set random seed.

    Args:
        seed (int): Seed to be used.
        deterministic (bool): Whether to set the deterministic option for
            CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
            to True and `torch.backends.cudnn.benchmark` to False.
            Default: False.
    """
    if seed is None:
        torch.backends.cudnn.benchmark = True
    else:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


def select_device(device="", apex=False, batch_size=None):
    import torch
    # device = 'cpu' or '0' or '0,1,2,3'
    cpu_request = device.lower() == "cpu"
    if device and not cpu_request:  # if device requested other than 'cpu'
        os.environ["CUDA_VISIBLE_DEVICES"] = device  # set environment variable
        assert torch.cuda.is_available(), (
            "CUDA unavailable, invalid device %s requested" % device
        )  # check availablity

    cuda = False if cpu_request else torch.cuda.is_available()
    if cuda:
        c = 1024**2  # bytes to MB
        ng = torch.cuda.device_count()
        if (
            ng > 1 and batch_size
        ):  # check that batch_size is compatible with device_count
            assert (
                batch_size % ng == 0
            ), "batch-size %g not multiple of GPU count %g" % (batch_size, ng)
        x = [torch.cuda.get_device_properties(i) for i in range(ng)]
        s = "Using CUDA " + (
            "Apex " if apex else ""
        )  # apex for mixed precision https://github.com/NVIDIA/apex
        for i in range(0, ng):
            if i == 1:
                s = " " * len(s)
            print(
                "%sdevice%g _CudaDeviceProperties(name='%s', total_memory=%dMB)"
                % (s, i, x[i].name, x[i].total_memory / c)
            )
    else:
        print("Using CPU")

    print("")  # skip a line
    return torch.device("cuda:0" if cuda else "cpu")


def write_config_log(logger, cfg=None, pkg_name=None):
    import torch
    import sys

    options = ""
    options += "==> torch version: {}\n".format(torch.__version__)
    options += "==> cudnn version: {}\n".format(torch.backends.cudnn.version())
    options += "==> Cmd:\n"
    options += str(sys.argv)

    if cfg:
        from gtgen.util.class_to_dict import ClassToDict

        cls2dict = ClassToDict(pkg_name, cfg)
        options += "\n==> config:\n"
        options += str(cls2dict)
        options += "\n"

    logger.info(options)

# def is_equal_struct(d1, d2):
#     if type(d1) != type(d2):
#         return False

#     if isinstance(d1, (dict, set)) and isinstance(d2, (dict, set)):
#         d1_keys = set(d1.keys())
#         d2_keys = set(d2.keys())
#         shared_keys = d1_keys.intersection(d2_keys)
#         if len(shared_keys)!=len(d1_keys) or len(shared_keys)!=len(d2_keys):
#             return False

#         if isinstance(d1, dict):
#             for key in shared_keys:
#                 if not is_equal_struct(d1[key], d2[key]):
#                     return False
#         return True
#     elif isinstance(d1, (list, tuple)):
#         if len(d1)!=len(d2):
#             return False
#         for idx in range(len(d1)):
#             if d1[idx] != d2[idx]:
#                 return False
#         return True
#     else:
#         return d1 == d2
