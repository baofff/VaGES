import core.lvm as lvm
import interface.datasets as datasets


lvm = {
    "class": lvm.GRBM,
    "kwargs": {
        "v_dim": "$(v_dim)",
        "h_dim": "$(h_dim)",
        "fix_std": False
    }
}


dataset = {
    "use_val": False,
    "class": datasets.Toy,
    "kwargs": {
        "data_type": "checkerboard",
    }
}
