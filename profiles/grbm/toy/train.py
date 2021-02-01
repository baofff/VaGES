
__all__ = ["train", "train_cd", "train_pcd", "train_ssm", "train_dsm", "train_mdsm",
           "train_bissm", "train_bidsm", "train_bimdsm", "train_vagesdsm",
           "train_ksd", "train_isksd", "train_vagesksd", "train_nce", "train_vnce"]


import core.criterions as criterions
from core.evaluate import reconstruct_error
import interface.evaluators as evaluators
from interface.utils import dict_utils
import core.inference.vi as vi
import torch.nn as nn
import profiles.common as common
from .base import lvm, dataset
import core.modules as modules
import develop.criterions.vages as vages
from develop.criterions.ksd import KSD, ISKSD


train = {
    "disable_ema": True,
    "disable_val_fn": True,
    "ckpt_root": "os.path.join($(workspace_root), 'train/ckpts/')",
    "v_dim": 2,
    "h_dim": 4,
    "training": {
        "n_ckpts": 10,
        "n_its": 100000,
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
                "ssm_score": {
                    "period": 500
                },
                "log_likelihood_score": {
                    "period": 500
                },
                "plot_sample_density": {
                    "period": 5000,
                    "kwargs": {
                        "left": 0.,
                        "right": 1.
                    }
                },
                "plot_sample_scatter": {
                    "period": 5000
                }
            }
        }
    },
    "interact": common.interact_datetime_train(period=100)
}


################################################################################
# The optimizers
################################################################################

adam_optimizer = common.adam_optimizer(lr=0.001)

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
# cd & pcd
################################################################################

train_cd = dict_utils.merge_dict(train, {
    "optimizers": {"all": adam_optimizer},
    "criterion": {
        "class": criterions.CD,
        "kwargs": {
            "k": 1
        }
    },
    "val_fn": {
        "fn": reconstruct_error,
    },
})

train_pcd = dict_utils.merge_dict(train, {
    "optimizers": {"all": adam_optimizer},
    "criterion": {
        "class": criterions.PCD,
        "kwargs": {
            "k": 1
        }
    },
    "val_fn": {
        "fn": reconstruct_error,
    },
})


################################################################################
# ssm
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
# dsm
################################################################################

dsm_kwargs = {
    "noise_std": 0.05
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

train_vagesdsm = dict_utils.merge_dict(train, {
    "optimizers": adam_optimizer_bilevel,
    "models": {
        "q": q
    },
    "criterion": {
        "class": vages.VaGESDSM,
        "kwargs": {
            **dsm_kwargs,
            "n_particles": 2,
            "n_lower_steps": 5,
            "lower_objective_type": "elbo",
            "sgld_type": None
        }
    },
})


################################################################################
# mdsm
################################################################################

mdsm_kwargs = {
    "sigma0": 0.05,
    "sigma_begin": 0.05,
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


################################################################################
# ksd
################################################################################

train_ksd = dict_utils.merge_dict(train, {
    "optimizers": {"all": adam_optimizer},
    "criterion": {
        "class": KSD,
    },
})

train_isksd = dict_utils.merge_dict(train, {
    "optimizers": {"all": adam_optimizer},
    "criterion": {
        "class": ISKSD,
        "kwargs": {
            "n_particles": 2,
        }
    },
})

train_vagesksd = dict_utils.merge_dict(train, {
    "optimizers": adam_optimizer_bilevel,
    "models": {
        "q": q
    },
    "criterion": {
        "class": vages.VaGESKSD,
        "kwargs": {
            "n_particles": 2,
            "n_lower_steps": 5,
            "lower_objective_type": "elbo",
            "sgld_type": None
        }
    },
})


################################################################################
# nce
################################################################################

nce_kwargs = {
    "nu": 1.0
}

c = {
    "class": modules.Const,
}

train_nce = dict_utils.merge_dict(train, {
    "models": {
        "c": c
    },
    "optimizers": {"all": adam_optimizer},
    "criterion": {
        "class": criterions.NCE,
        "kwargs": {**nce_kwargs}
    },
})

train_vnce = dict_utils.merge_dict(train, {
    "models": {
        "c": c,
        "q": q
    },
    "optimizers": {
        "lower": {
            **adam_optimizer,
            "model_keys": ["q"]
        },
        "higher": {
            **adam_optimizer,
            "model_keys": ["lvm", "c"]
        }
    },
    "criterion": {
        "class": criterions.VNCE,
        "kwargs": {
            **nce_kwargs,
            "n_particles": 5,
            "n_lower_steps": 5
        }
    },
})
