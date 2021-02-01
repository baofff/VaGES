from interface.runner import run_train_profile
import profiles.grbm.toy as profiles
from interface.utils import dict_utils, task_schedule
from multiprocessing import Process
from interface.utils.task_schedule import Task
import datetime
import os


def train():
    now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    prefix = "workspace/runner/grbm/toy_ksd/{}".format(now)

    tasks = []

    for seed in [1, 22, 77, 123, 666, 1234, 3333, 7777, 9090, 23333]:
        # isksd
        for n_particles in [2, 5, 10]:
            profile = dict_utils.merge_dict(profiles.train_isksd, {
                "seed": seed,
                "workspace_root": os.path.join(prefix, "isksd%d_seed%d" % (n_particles, seed)),
                "criterion": {
                    "kwargs": {
                        "n_particles": n_particles
                    }
                }
            })
            p = Process(target=run_train_profile, args=(profile,))
            tasks.append(Task(p, 1))

        # vagesksd
        for n_particles in [2, 5, 10]:
            profile = dict_utils.merge_dict(profiles.train_vagesksd, {
                "seed": seed,
                "workspace_root": os.path.join(prefix, "vagesksd%d_seed%d" % (n_particles, seed)),
                "criterion": {
                    "kwargs": {
                        "n_particles": n_particles
                    }
                }
            })
            p = Process(target=run_train_profile, args=(profile,))
            tasks.append(Task(p, 1))

        # ksd
        profile = dict_utils.merge_dict(profiles.train_ksd, {
            "seed": seed,
            "workspace_root": os.path.join(prefix, "ksd_seed%d" % seed)
        })
        p = Process(target=run_train_profile, args=(profile,))
        tasks.append(Task(p, 1))

    task_schedule.wait_schedule(tasks, devices=task_schedule.available_devices())


if __name__ == "__main__":
    train()
