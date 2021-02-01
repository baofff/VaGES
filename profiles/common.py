import torch.optim as optim


################################################################################
# The commonly used interact
################################################################################

def interact_datetime_train(period: int):
    return {
        "fname_log": "os.path.join($(workspace_root), 'train/logs/" +
                     "{}.log'.format(datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')))",
        "summary_root": "os.path.join($(workspace_root), 'train/summary/')",
        "period": period,
    }


def interact_datetime_evaluate():
    return {
        "fname_log": "os.path.join($(workspace_root), 'evaluate/logs/" +
                     "{}.log'.format(datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')))",
        "summary_root": "os.path.join($(workspace_root), 'evaluate/summary/')",
    }


################################################################################
# The commonly used optimizer and lr_scheduler
################################################################################

def adam_optimizer(lr: float):
    return {
        "class": optim.Adam,
        "kwargs": {
            "lr": lr,
            "betas": (0.9, 0.95)
        }
    }


def cosine_lr_scheduler():
    return {
        "class": optim.lr_scheduler.CosineAnnealingLR,
        "kwargs": {
            "T_max": "$(training.n_its)",
            "eta_min": 1e-6
        }
    }

