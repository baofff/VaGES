from interface.runner import run_train_profile, run_evaluate_profile
from profiles.eblvm.cifar10.train_dev import train_vagesmdsm
from profiles.eblvm.cifar10.evaluate import evaluate
from interface.utils.exp_templates import run_on_different_settings, evaluate_one_ckpt


def run_on_different_setting_train(names, settings, prefix="vagesmdsm", n_devices=1, common_setting=None, devices=None, time=None):
    run_on_different_settings(run_fn=run_train_profile, profile=train_vagesmdsm, path="workspace/runner/eblvm/cifar10",
                              prefix=prefix, names=names, settings=settings,
                              n_devices=n_devices, common_setting=common_setting,
                              devices=devices, time=time, time_strategy="now")


def run_on_different_setting_evaluate(names, ckpts, settings=None, prefix="vagesmdsm", n_devices=1, common_setting=None, devices=None, time=None):
    evaluate_one_ckpt(run_fn=run_evaluate_profile, profile=evaluate, path="workspace/runner/eblvm/cifar10",
                      prefix=prefix, names=names, ckpts=ckpts, n_devices=n_devices,
                      settings=settings, common_setting=common_setting, devices=devices, time=time)


if __name__ == "__main__":
    tag = "train"

    if tag == "train":
        common = {"criterion": {"kwargs": {"sgld_steps": 5, "vr": False}}, "h_dim": 50}
        _settings = {}
        for alpha in ["1e-2"]:
            for sigma in ["1e-4"]:
                _key = "alpha_%s_sigma_%s" % (alpha, sigma)
                _settings[_key] = {"criterion": {"kwargs": {"alpha": float(alpha), "sigma": float(sigma)}}}
        run_on_different_setting_train(list(_settings.keys()), list(_settings.values()),
                                       prefix="vagesmdsm_tune_sgld_hdim50", common_setting=common)

    elif tag == "evaluate":
        common = {"h_dim": 50}
        _settings = {
            "alpha_1e-2_sigma_1e-4": "200000.ckpt.pth",
        }
        run_on_different_setting_evaluate(names=list(_settings.keys()), ckpts=list(_settings.values()),
                                          prefix="vagesmdsm_tune_sgld_hdim50", common_setting=common)

    # sensitivity analysis below
    elif tag == "elbo":
        common = {"criterion": {"kwargs": {"sgld_steps": 5, "vr": False, "alpha": 1e-2, "sigma": 1e-4, "lower_objective_type": "elbo"}}, "h_dim": 50}
        _settings = {
            "elbo": {},
        }
        run_on_different_setting_train(list(_settings.keys()), list(_settings.values()), prefix="elbo", common_setting=common)

    elif tag == "n_particles":
        common = {"criterion": {"kwargs": {"sgld_steps": 5, "vr": False, "alpha": 1e-2, "sigma": 1e-4}}, "h_dim": 50}
        _settings = {
            "n_particles_5": {"criterion": {"kwargs": {"n_particles": 5}}},
            "n_particles_10": {"criterion": {"kwargs": {"n_particles": 10}}},
        }
        run_on_different_setting_train(names=list(_settings.keys()), settings=list(_settings.values()),
                                       prefix="n_particles", common_setting=common)

    elif tag == "res_structure":
        common = {"criterion": {"kwargs": {"sgld_steps": 5, "vr": False, "alpha": 1e-2, "sigma": 1e-4}}, "h_dim": 50}
        _settings = {
            "k_2": {"models": {"lvm": {"kwargs": {"feature_net": {"kwargs": {"k": 2}}}}}},
            "k_4": {"models": {"lvm": {"kwargs": {"feature_net": {"kwargs": {"k": 4}}}}}},
        }
        run_on_different_setting_train(list(_settings.keys()), list(_settings.values()),
                                       prefix="res_structure", common_setting=common)

    elif tag == "n_lower_steps_sgld_steps":
        common = {"criterion": {"kwargs": {"vr": False, "alpha": 1e-2, "sigma": 1e-4}}, "h_dim": 50}
        _settings = {
            "n_lower_steps_0_sgld_steps_0": {"criterion": {"kwargs": {"n_lower_steps": 0, "sgld_steps": 0}}},
            "n_lower_steps_5_sgld_steps_0": {"criterion": {"kwargs": {"n_lower_steps": 5, "sgld_steps": 0}}},
            "n_lower_steps_0_sgld_steps_5": {"criterion": {"kwargs": {"n_lower_steps": 0, "sgld_steps": 5}}},
            "n_lower_steps_0_sgld_steps_10": {"criterion": {"kwargs": {"n_lower_steps": 0, "sgld_steps": 10}}},
            "n_lower_steps_5_sgld_steps_5": {"criterion": {"kwargs": {"n_lower_steps": 5, "sgld_steps": 5}}},
            "n_lower_steps_10_sgld_steps_0": {"criterion": {"kwargs": {"n_lower_steps": 10, "sgld_steps": 0}}},
            "n_lower_steps_0_sgld_steps_15": {"criterion": {"kwargs": {"n_lower_steps": 0, "sgld_steps": 15}}},
            "n_lower_steps_5_sgld_steps_10": {"criterion": {"kwargs": {"n_lower_steps": 5, "sgld_steps": 10}}},
            "n_lower_steps_10_sgld_steps_5": {"criterion": {"kwargs": {"n_lower_steps": 10, "sgld_steps": 5}}},
            "n_lower_steps_15_sgld_steps_0": {"criterion": {"kwargs": {"n_lower_steps": 15, "sgld_steps": 0}}},
        }
        run_on_different_setting_train(names=list(_settings.keys()), settings=list(_settings.values()),
                                       prefix="n_lower_steps_sgld_steps", common_setting=common)
