import core.lvm as lvm
import core.modules as modules
import interface.datasets as datasets
import core.inference.vi.param_nets as param_nets
import core.inference.vi as vi


lvm = {
    "class": lvm.EBLVMACP,
    "kwargs": {
        "v_shape": "$(v_shape)",
        "h_dim": "$(h_dim)",
        "feature_net": {
            "class": modules.ResidualNetDown,
            "kwargs": {
                "in_channels": "$(v_shape)[0]",
                "channels": 128,
                "k": 3
            }
        },
        "scalar_net": {
            "class": modules.LinearAFSquare,
            "kwargs": {
                "in_features": "$(h_dim) * 2",
                "features": "($(v_shape)[1] // 8) ** 2 * 4 *"
                            "$(models.lvm.kwargs.feature_net.kwargs.channels)",
            }
        },
    }
}


q = {
    "class": vi.AmortizedGauss,
    "kwargs": {
        "param_net": {
            "class": param_nets.ConvParamNet,
            "kwargs": {
                "v_shape": "$(v_shape)",
                "h_dim": "$(h_dim)",
                "k": 1,
            }
        },
        "h_dim": "$(h_dim)",
    }
}


dataset = {
    "use_val": False,
    "class": datasets.CelebA,
    "kwargs": {
        "data_path": "workspace/datasets/celeba/",
        "width": "$(v_shape)[1]"
    }
}
