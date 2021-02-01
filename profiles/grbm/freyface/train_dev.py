
__all__ = ["train_vagesdsm"]


from interface.utils import dict_utils
from .train import train, adam_optimizer_bilevel, q, dsm_kwargs
from develop.criterions import vages


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
