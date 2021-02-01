
__all__ = ["train", "train_bimdsm"]


import core.criterions as criterions
from interface.utils import dict_utils
import profiles.common as common
import interface.evaluators as evaluators
from .base import lvm, q, dataset


train = {
    "ckpt_root": "os.path.join($(workspace_root), 'train/ckpts/')",
    "v_shape": [3, 64, 64],
    "h_dim": 50,
    "training": {
        "n_ckpts": 60,
        "n_its": 300000,
        "batch_size": 100,
    },
    "models": {
        "lvm": lvm,
    },
    "dataset": dataset,
    "interact": common.interact_datetime_train(period=10),
    "evaluator": {
        "class": evaluators.EBLVMEvaluator,
        "kwargs": {
            "evaluator_root": "os.path.join($(workspace_root), 'train/evaluator/')",
            "options": {
                "grid_sample": {
                    "period": 15000
                },
            }
        }
    },
}


################################################################################
# The optimizers and lr_schedulers
################################################################################

adam_optimizer = common.adam_optimizer(lr=0.00005)

adam_optimizer_bilevel = {
    "lower": {
        **adam_optimizer,
        "model_keys": ["q"]
    },
    "higher": {
        **adam_optimizer,
        "model_keys": ["lvm"]
    }
}

lr_scheduler_bilevel = {
    "lower": common.cosine_lr_scheduler(),
    "higher": common.cosine_lr_scheduler()
}


################################################################################
# bimdsm
################################################################################

train_bimdsm = dict_utils.merge_dict(train, {
    "optimizers": adam_optimizer_bilevel,
    "lr_schedulers": lr_scheduler_bilevel,
    "models": {
        "q": q
    },
    "criterion": {
        "class": criterions.BiMDSM,
        "kwargs": {
            "sigma0": 0.1,
            "sigma_begin": 0.05,
            "sigma_end": 1.2,
            "dist": "linear",
            "n_lower_steps": 5,
            "n_unroll": 0,
            "lower_objective_type": "posterior_fisher"
        }
    },
})
