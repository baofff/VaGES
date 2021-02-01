import core.lvm as lvm
import core.modules as modules
import core.inference.vi as vi
import core.inference.vi.param_nets as param_nets
import interface.datasets as datasets


lvm = {
    "class": lvm.EBLVMACP,
    "kwargs": {
        "v_dim": 2,
        "h_dim": "$(h_dim)",
        "feature_net": {
            "class": modules.MLPResidualNet,
            "kwargs": {
                "n_features_lst": [2, 100, 100, 100],
            }
        },
        "scalar_net": {
            "class": modules.LinearAFSquare,
            "kwargs": {
                "in_features": "$(h_dim) * 2",
                "features": 100,
            }
        },
    }
}


q = {
    "class": vi.AmortizedGauss,
    "kwargs": {
        "param_net": {
            "class": param_nets.MLPResidualParamNet,
            "kwargs": {
                "v_dim": 2,
                "h_dim": "$(h_dim)",
                "n_hiddens_lst": [100, 100, 100],
            }
        },
        "h_dim": "$(h_dim)",
    }
}


dataset = {
    "use_val": False,
    "class": datasets.Toy,
    "kwargs": {
        "data_type": "checkerboard",
    }
}
