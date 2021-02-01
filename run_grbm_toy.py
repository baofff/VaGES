from interface.runner import run_train_profile
import profiles.grbm.toy as profiles
from interface.utils import dict_utils, task_schedule
from multiprocessing import Process
from interface.utils.task_schedule import Task
import datetime
import os


def train():
    now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    prefix = "workspace/runner/grbm/toy/{}".format(now)

    tasks = []

    for seed in [1, 22, 77, 123, 666, 1234, 3333, 7777, 9090, 23333]:
        profile = dict_utils.merge_dict(profiles.train_cd, {
            "seed": seed,
            "workspace_root": os.path.join(prefix, "cd1_seed%d" % seed),
            "criterion": {
                "kwargs": {
                    "k": 1
                }
            }
        })
        p = Process(target=run_train_profile, args=(profile,))
        tasks.append(Task(p, 1))

        profile = dict_utils.merge_dict(profiles.train_pcd, {
            "seed": seed,
            "workspace_root": os.path.join(prefix, "pcd1_seed%d" % seed),
            "criterion": {
                "kwargs": {
                    "k": 1
                }
            }
        })
        p = Process(target=run_train_profile, args=(profile,))
        tasks.append(Task(p, 1))

        lst = [(profiles.train_dsm, "dsm_seed%d" % seed),
               (profiles.train_bidsm, "bidsm_seed%d" % seed),
               (profiles.train_vagesdsm, "vagesdsm_seed%d" % seed),
               (profiles.train_nce, "nce_seed%d" % seed), (profiles.train_vnce, "vnce_seed%d" % seed)]

        for pf, name in lst:
            profile = dict_utils.merge_dict(pf, {
                "seed": seed,
                "workspace_root": os.path.join(prefix, name)
            })
            p = Process(target=run_train_profile, args=(profile,))
            tasks.append(Task(p, 1))

    task_schedule.wait_schedule(tasks, devices=task_schedule.available_devices())


if __name__ == "__main__":
    train()
