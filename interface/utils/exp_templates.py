
__all__ = ["run_on_different_settings", "run_one_setting", "run_ablation_study",
           "sample_last_ckpts", "sample_last_ckpts_after_ablation_study_train",
           "sample_from_one_ckpt", "sample_from_one_ckpt_after_ablation_study_train", "evaluate_one_ckpt"]


from .misc import get_root_by_time
from .dict_utils import merge_dict, single_chain_dict
import os
from multiprocessing import Process
from .task_schedule import Task, wait_schedule, available_devices
from typing import List, Union
from .ckpt import list_ckpts


def run_on_different_settings(run_fn, profile: dict, path: str, prefix: str, names: List[str], settings: List[dict],
                              n_devices: Union[int, List[int]] = 1, common_setting: dict = None,
                              devices=None, time: str = None, time_strategy: str = None):
    r"""
    Args:
        run_fn: the running function
        profile: the profile template
        path: the result of each experiment will be saved in "path/prefix_time/name"
        prefix: the result of each experiment will be saved in "path/prefix_time/name"
        names: the result of each experiment will be saved in "path/prefix_time/name"
        settings: the settings of experiments
        n_devices: the number of devices for each experiment
        common_setting: the common experimental setting
        devices: the devices to use
        time: a time tag with the format %Y-%m-%d-%H-%M-%S
        time_strategy: how to infer the time when time is None (only works when time is None )
    """
    assert len(settings) == len(names)
    if not isinstance(n_devices, int):
        assert len(settings) == len(n_devices)
    path_prefix_time = get_root_by_time(path, prefix, time, time_strategy)

    tasks = []
    if isinstance(n_devices, int):
        n_devices = [n_devices] * len(settings)
    for setting, name, n_device in zip(settings, names, n_devices):
        _profile = profile
        if common_setting is not None:
            _profile = merge_dict(_profile, common_setting)
        _profile = merge_dict(_profile, {
            "workspace_root": os.path.join(path_prefix_time, name),
            **setting
        })
        p = Process(target=run_fn, args=(_profile,))
        tasks.append(Task(p, n_device))

    if devices is None:
        devices = available_devices()
    wait_schedule(tasks, devices=devices)


def run_one_setting(run_fn, profile: dict, path: str, prefix: str, name: str, setting: dict = None,
                    n_device: int = 1, devices=None, time: str = None, time_strategy: str = None):
    if setting is None:
        setting = {}
    run_on_different_settings(run_fn=run_fn, profile=profile, path=path, prefix=prefix, names=[name],
                              settings=[setting], n_devices=n_device, devices=devices, time=time,
                              time_strategy=time_strategy)


def run_ablation_study(run_fn, profile: dict, path: str, prefix: str, key: str, vals: List,
                       n_devices: Union[int, List[int]] = 1, common_setting=None, devices=None,
                       time: str = None, time_strategy: str = None):
    r""" The result will be saved in "path/prefix_time"
        The result of each experiment will be saved in "path/prefix_time/key_val"
    Args:
        run_fn: the running function
        profile: the profile template
        path: the result of each experiment will be saved in "path/prefix_time/key_val"
        prefix: the result of each experiment will be saved in "path/prefix_time/key_val"
        key: the key of the variable studied
        vals: the values studied
        n_devices: the number of devices for each experiment
        common_setting: the common experimental setting
        devices: the devices to use
        time: a time tag with the format %Y-%m-%d-%H-%M-%S
        time_strategy: how to infer the time when time is None (only works when time is None )
    """
    settings, names = [], []
    for val in vals:
        settings.append(single_chain_dict(key, val))
        names.append("{}_{}".format(key, val))

    run_on_different_settings(run_fn=run_fn, profile=profile, path=path, prefix=prefix, settings=settings,
                              names=names, n_devices=n_devices, common_setting=common_setting,
                              devices=devices, time=time, time_strategy=time_strategy)


def sample_last_ckpts(run_fn, profile: dict, path: str, prefix: str, names: List[str],
                      n_ckpts: int, n_samples: int, batch_size: int, n_devices: Union[int, List[int]] = 1,
                      settings: List[dict] = None, common_setting: dict = None,
                      devices=None, time: str = None):
    r""" 1000 samples for validation
    """
    if settings is None:
        settings = [{} for _ in names]
    assert len(settings) == len(names)
    if not isinstance(n_devices, int):
        assert len(settings) == len(n_devices)
    path_prefix_time = get_root_by_time(path, prefix, time, "latest")

    tasks = []
    if isinstance(n_devices, int):
        n_devices = [n_devices] * len(settings)

    for setting, name, n_device in zip(settings, names, n_devices):
        workspace_root = os.path.join(path_prefix_time, name)
        ckpt_root = os.path.join(workspace_root, 'train/ckpts/')
        fnames = list_ckpts(ckpt_root)[-n_ckpts:]
        for fname in fnames:
            _profile = profile
            if common_setting is not None:
                _profile = merge_dict(_profile, common_setting)
            _profile = merge_dict(_profile, {
                "path": os.path.join(workspace_root,
                                     "evaluate/evaluator/sample2dir/{}/sample{}".format(fname, n_samples)),
                "n_samples": n_samples,
                "batch_size": batch_size,
                "workspace_root": workspace_root,
                "ckpt_path": os.path.join(ckpt_root, fname),
                **setting,
            })
            p = Process(target=run_fn, args=(_profile,))
            tasks.append(Task(p, n_device))

    if devices is None:
        devices = available_devices()
    wait_schedule(tasks, devices=devices)


def sample_last_ckpts_after_ablation_study_train(
        run_fn, profile: dict, path: str, prefix: str, key: str, vals: List,
        n_ckpts: int, n_samples: int, batch_size: int,
        n_devices: Union[int, List[int]] = 1, common_setting=None, devices=None, time: str = None):
    settings, names = [], []
    for val in vals:
        settings.append(single_chain_dict(key, val))
        names.append("{}_{}".format(key, val))
    sample_last_ckpts(run_fn=run_fn, profile=profile, path=path, prefix=prefix, names=names, settings=settings,
                      n_ckpts=n_ckpts, n_samples=n_samples, batch_size=batch_size,
                      n_devices=n_devices, common_setting=common_setting,
                      devices=devices, time=time)


def sample_from_one_ckpt(run_fn, profile: dict, path: str, prefix: str, names: List[str],
                         ckpts: List[str], n_samples: int, batch_size: int, n_devices: Union[int, List[int]] = 1,
                         settings: List[dict] = None, common_setting: dict = None,
                         devices=None, time: str = None):
    if settings is None:
        settings = [{} for _ in names]
    assert len(names) == len(settings)
    assert len(names) == len(ckpts)
    _settings = []
    for setting, name, ckpt in zip(settings, names, ckpts):
        path_prefix_time = get_root_by_time(path, prefix, time, "latest")
        workspace_root = os.path.join(path_prefix_time, name)
        ckpt_root = os.path.join(workspace_root, 'train/ckpts/')
        _settings.append(merge_dict(setting, {
            "n_samples": n_samples,
            "batch_size": batch_size,
            "ckpt_path": os.path.join(ckpt_root, ckpt),
            "path": os.path.join(workspace_root, "evaluate/evaluator/sample2dir/{}/sample50000".format(ckpt)),
        }))

    run_on_different_settings(run_fn=run_fn, profile=profile, path=path, prefix=prefix, names=names, settings=_settings,
                              n_devices=n_devices, common_setting=common_setting,
                              devices=devices, time=time, time_strategy="latest")


def sample_from_one_ckpt_after_ablation_study_train(
        run_fn, profile: dict, path: str, prefix: str, key: str, vals: List,
        ckpts: List[str], n_samples: int, batch_size: int,
        n_devices: Union[int, List[int]] = 1, common_setting=None, devices=None, time: str = None):
    settings, names = [], []
    for val in vals:
        settings.append(single_chain_dict(key, val))
        names.append("{}_{}".format(key, val))
    sample_from_one_ckpt(run_fn=run_fn, profile=profile, path=path, prefix=prefix, names=names, settings=settings,
                         ckpts=ckpts, n_samples=n_samples, batch_size=batch_size,
                         n_devices=n_devices, common_setting=common_setting,
                         devices=devices, time=time)


def evaluate_one_ckpt(run_fn, profile: dict, path: str, prefix: str, names: List[str],
                      ckpts: List[str], n_devices: Union[int, List[int]] = 1,
                      settings: List[dict] = None, common_setting: dict = None,
                      devices=None, time: str = None):
    if settings is None:
        settings = [{} for _ in names]
    assert len(names) == len(settings)
    assert len(names) == len(ckpts)
    _settings = []
    for setting, name, ckpt in zip(settings, names, ckpts):
        path_prefix_time = get_root_by_time(path, prefix, time, "latest")
        workspace_root = os.path.join(path_prefix_time, name)
        ckpt_root = os.path.join(workspace_root, 'train/ckpts/')
        _settings.append(merge_dict(setting, {
            "ckpt_path": os.path.join(ckpt_root, ckpt),
        }))

    run_on_different_settings(run_fn=run_fn, profile=profile, path=path, prefix=prefix, names=names, settings=_settings,
                              n_devices=n_devices, common_setting=common_setting,
                              devices=devices, time=time, time_strategy="latest")
