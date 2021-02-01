
__all__ = ["evaluate"]


import core.lvm as lvm
import interface.datasets as datasets
import interface.evaluators as evaluators
import profiles.common as common


evaluate = {
    "summary_root": "os.path.join($(workspace_root), 'evaluate/summary/')",
    "v_dim": 560,
    "h_dim": 400,
    "models": {
        "lvm": {
            "class": lvm.GRBM,
            "kwargs": {
                "v_dim": "$(v_dim)",
                "h_dim": "$(h_dim)",
                "fix_std": False
            }
        },
    },
    "dataset": {
        "class": datasets.FreyFace,
        "kwargs": {
            "data_path": "workspace/datasets/freyface/",
            "gauss_noise": False
        }
    },
    "evaluator": {
        "class": evaluators.RBMEvaluator,
        "kwargs": {
            "evaluator_root": "os.path.join($(workspace_root), 'evaluate/evaluator/')",
            "options": {
                "grid_sample": {},
                "ssm_score": {
                    "kwargs": {
                        "batch_size": 100,
                    }
                },
                "log_likelihood_score": {
                    "kwargs": {
                        "batch_size": 100,
                    }
                }
            }
        }
    },
    "interact": common.interact_datetime_evaluate()
}
