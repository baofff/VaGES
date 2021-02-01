
__all__ = ["train", "train_bimdsm", "train_vagesmdsm"]


import core.criterions as criterions
from interface.utils import dict_utils
import profiles.common as common
import interface.evaluators as evaluators
from .base import lvm, q, dataset
from develop.criterions.vages import VaGESMDSM


train = {
    "ckpt_root": "os.path.join($(workspace_root), 'train/ckpts/')",
    "h_dim": 10,
    "training": {
        "n_ckpts": 10,
        "n_its": 100000,
        "batch_size": 100,
    },
    "models": {
        "lvm": lvm
    },
    "dataset": dataset,
    "interact": common.interact_datetime_train(period=100),
    "evaluator": {
        "class": evaluators.EBLVMEvaluator,
        "kwargs": {
            "evaluator_root": "os.path.join($(workspace_root), 'train/evaluator/')",
            "options": {
                "plot_sample_scatter": {
                    "period": 5000
                }
            }
        }
    },
}


################################################################################
# The optimizers and lr_schedulers
################################################################################

adam_optimizer = common.adam_optimizer(lr=0.0001)

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


mdsm_kwargs = {
    "sigma0": 0.05,
    "sigma_begin": 0.05,
    "sigma_end": 1.0,
    "dist": "geometrical",
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
            **mdsm_kwargs,
            "n_lower_steps": 5,
            "n_unroll": 0,
            "lower_objective_type": "posterior_fisher"
        }
    },
})


################################################################################
# vagesmdsm
################################################################################

train_vagesmdsm = dict_utils.merge_dict(train, {
    "disable_val_fn": True,
    "optimizers": adam_optimizer_bilevel,
    "lr_schedulers": lr_scheduler_bilevel,
    "models": {
        "q": q
    },
    "criterion": {
        "class": VaGESMDSM,
        "kwargs": {
            **mdsm_kwargs,
            "n_particles": 2,
            "n_lower_steps": 5,
            "lower_objective_type": "posterior_fisher"
        }
    },
})
