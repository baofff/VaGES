
__all__ = ["evaluate", "sample2dir"]


import interface.evaluators as evaluators
from .base import lvm, q, dataset
import profiles.common as common


evaluate = {
    "v_shape": [1, 32, 32],
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
                "deterministic_interpolate_linear": {"kwargs": {"ratio": 1.}},
            }
        }
    },
    "interact": common.interact_datetime_evaluate()
}


sample2dir = {
    "v_shape": [1, 32, 32],
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
