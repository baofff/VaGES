from interface.utils import dict_utils
from develop.criterions.vages import VaGESMDSM
from .train import train, adam_optimizer_bilevel, lr_scheduler_bilevel, q


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
            "sigma0": 0.1,
            "sigma_begin": 0.05,
            "sigma_end": 1.2,
            "dist": "linear",
            "n_particles": 2,
            "n_lower_steps": 5,
            "lower_objective_type": "posterior_fisher"
        }
    },
})
