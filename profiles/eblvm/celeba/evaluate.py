
__all__ = ["sample2dir"]


import interface.evaluators as evaluators
import profiles.common as common
from .base import lvm, q, dataset


evaluate = {
    "v_shape": "$(v_shape)",
    "models": {
        "lvm": lvm,
        "q": q
    },
    "dataset": dataset,
    "evaluator": {
        "class": evaluators.EBLVMEvaluator,
        "kwargs": {
            "evaluator_root": "os.path.join($(workspace_root), 'evaluate/evaluator/')",
            "options": {
                "grid_sample": {},
                "deterministic_interpolate_linear": {"kwargs": {"ratio": 3.}},
            }
        }
    },
    "interact": common.interact_datetime_evaluate()
}


sample2dir = {
    "v_shape": "$(v_shape)",
    "models": {
        "lvm": lvm,
        "q": q
    },
    "dataset": dataset,
    "evaluator": {
        "class": evaluators.EBLVMEvaluator,
        "kwargs": {
            "evaluator_root": "os.path.join($(workspace_root), 'evaluate/evaluator/')",
            "options": {
                "sample2dir": {
                    "kwargs": {
                        "path": "$(path)",
                        "n_samples": "$(n_samples)",
                        "batch_size": "$(batch_size)"
                    }
                },
            }
        }
    },
}


sample2dir_traverse_ckpts = {
    "ckpt_root": "os.path.join($(workspace_root), 'train/ckpts/')",
    "v_shape": "$(v_shape)",
    "models": {
        "lvm": lvm,
        "q": q
    },
    "dataset": dataset,
    "interact": common.interact_datetime_evaluate(),
    "evaluator": {
        "class": evaluators.EBLVMEvaluator,
        "kwargs": {
            "evaluator_root": "os.path.join($(workspace_root), 'evaluate/evaluator/')",
            "options": {
                "sample2dir": {
                    "kwargs": {
                        "n_samples": "$(n_samples)",
                        "batch_size": "$(batch_size)"
                    }
                },
            }
        }
    },
}
