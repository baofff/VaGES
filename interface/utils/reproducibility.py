
__all__ = ["set_seed", "set_deterministic", "backup_codes", "backup_profile"]


import torch
import numpy as np
import os
import datetime
import shutil
import pprint


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)


def set_deterministic(flag: bool):
    if flag:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def backup_codes(path):
    current_path = os.path.dirname(os.path.realpath(__file__))
    root_path = os.path.realpath(os.path.join(current_path, os.pardir, os.pardir))

    path = os.path.join(path, "codes_{}".format(datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")))
    os.makedirs(path, exist_ok=True)

    names = ["core", "interface", "profiles", "develop", "experiments", "tools"]
    for name in names:
        if os.path.exists(os.path.join(root_path, name)):
            shutil.copytree(os.path.join(root_path, name), os.path.join(path, name))

    pyfiles = filter(lambda x: x.endswith(".py"), os.listdir(root_path))
    for pyfile in pyfiles:
        shutil.copy(os.path.join(root_path, pyfile), os.path.join(path, pyfile))


def backup_profile(profile: dict, path):
    os.makedirs(path, exist_ok=True)
    path = os.path.join(path, "profile_{}.txt".format(datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")))
    s = pprint.pformat(profile)
    with open(path, 'w') as f:
        f.write(s)
