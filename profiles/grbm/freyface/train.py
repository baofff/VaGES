
__all__ = ["train", "train_cd", "train_ssm", "train_dsm", "train_mdsm",
           "train_bissm", "train_bidsm", "train_bimdsm"]


import core.criterions as criterions
from core.evaluate import negative_log_likelihood_score_4_grbm
import interface.evaluators as evaluators
from interface.utils import dict_utils
import core.inference.vi as vi
import torch.nn as nn
import profiles.common as common
from .base import lvm, dataset


train = {
    "disable_ema": True,
    "ckpt_root": "os.path.join($(workspace_root), 'train/ckpts/')",
    "v_dim": 560,
    "h_dim": 400,
    "training": {
        "n_ckpts": 10,
        "n_its": 20000,
        "batch_size": 100,
    },
    "models": {
        "lvm": lvm,
    },
    "dataset": dataset,
    "evaluator": {
        "class": evaluators.RBMEvaluator,
        "kwargs": {
            "evaluator_root": "os.path.join($(workspace_root), 'train/evaluator')",
            "options": {
                "grid_sample": {
                    "period": "$(training.n_its) // (2 * $(training.n_ckpts))"
                },
            }
        }
    },
    "interact": common.interact_datetime_train(period=100),
    "val_fn": {
        "apply_to": "dataset",
        "fn": negative_log_likelihood_score_4_grbm,
    },
}


################################################################################
# The optimizers
################################################################################

adam_optimizer = common.adam_optimizer(lr=0.0002)

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


################################################################################
# The variational posterior
################################################################################

q = {
    "class": vi.AmortizedBernoulli,
    "kwargs": {
        "param_net": {
            "class": nn.Linear,
            "kwargs": {
                "in_features": "$(v_dim)",
                "out_features": "$(h_dim)"
            }
        },
        "h_dim": "$(h_dim)",
    }
}


################################################################################
# cd
################################################################################

train_cd = dict_utils.merge_dict(train, {
    "optimizers": {"all": adam_optimizer},
    "criterion": {
        "class": criterions.CD,
        "kwargs": {
            "k": 1
        }
    },
})


################################################################################
# ssm & bissm
################################################################################

train_ssm = dict_utils.merge_dict(train, {
    "optimizers": {"all": adam_optimizer},
    "criterion": {
        "class": criterions.SSM,
    },
})

train_bissm = dict_utils.merge_dict(train, {
    "optimizers": adam_optimizer_bilevel,
    "models": {
        "q": q
    },
    "criterion": {
        "class": criterions.BiSSM,
        "kwargs": {
            "n_lower_steps": 5,
            "n_unroll": 5,
            "lower_objective_type": "elbo"
        }
    },
})


################################################################################
# dsm & bidsm
################################################################################

dsm_kwargs = {
    "noise_std": 0.3
}

train_dsm = dict_utils.merge_dict(train, {
    "optimizers": {"all": adam_optimizer},
    "criterion": {
        "class": criterions.DSM,
        "kwargs": {**dsm_kwargs}
    },
})

train_bidsm = dict_utils.merge_dict(train, {
    "optimizers": adam_optimizer_bilevel,
    "models": {
        "q": q
    },
    "criterion": {
        "class": criterions.BiDSM,
        "kwargs": {
            **dsm_kwargs,
            "n_lower_steps": 5,
            "n_unroll": 5,
            "lower_objective_type": "elbo"
        }
    },
})


################################################################################
# mdsm & bimdsm
################################################################################

mdsm_kwargs = {
    "sigma0": 0.3,
    "sigma_begin": 0.3,
    "sigma_end": 1.0,
    "dist": "geometrical"
}


train_mdsm = dict_utils.merge_dict(train, {
    "optimizers": {"all": adam_optimizer},
    "criterion": {
        "class": criterions.MDSM,
        "kwargs": {**mdsm_kwargs}
    },
})


train_bimdsm = dict_utils.merge_dict(train, {
    "optimizers": adam_optimizer_bilevel,
    "models": {
        "q": q
    },
    "criterion": {
        "class": criterions.BiMDSM,
        "kwargs": {
            **mdsm_kwargs,
            "n_lower_steps": 5,
            "n_unroll": 5,
            "lower_objective_type": "elbo"
        }
    },
})
