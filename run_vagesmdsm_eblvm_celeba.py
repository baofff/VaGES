from interface.runner import run_train_profile, run_evaluate_profile
from profiles.eblvm.celeba.train_dev import train_vagesmdsm
from profiles.eblvm.celeba.evaluate import evaluate
from interface.utils.exp_templates import run_on_different_settings, evaluate_one_ckpt


def run_on_different_setting_train(names, settings, prefix="vagesmdsm", n_devices=1, common_setting=None, devices=None, time=None):
    run_on_different_settings(run_fn=run_train_profile, profile=train_vagesmdsm, path="workspace/runner/eblvm/celeba",
                              prefix=prefix, names=names, settings=settings,
                              n_devices=n_devices, common_setting=common_setting,
                              devices=devices, time=time, time_strategy="now")


def run_on_different_setting_evaluate(names, ckpts, settings=None, prefix="vagesmdsm", n_devices=1, common_setting=None, devices=None, time=None):
    evaluate_one_ckpt(run_fn=run_evaluate_profile, profile=evaluate, path="workspace/runner/eblvm/celeba",
                      prefix=prefix, names=names, ckpts=ckpts, n_devices=n_devices,
                      settings=settings, common_setting=common_setting, devices=devices, time=time)


if __name__ == "__main__":
    tag = "train"

    if tag == "train":
        common = {"criterion": {"kwargs": {"sgld_steps": 50, "vr": False}}, "v_shape": [3, 64, 64], "h_dim": 50}
        _settings = {}
        for alpha in ["5e-4"]:
            for sigma in ["1e-4"]:
                _key = "alpha_%s_sigma_%s" % (alpha, sigma)
                _settings[_key] = {"criterion": {"kwargs": {"alpha": float(alpha), "sigma": float(sigma)}}}

        run_on_different_setting_train(list(_settings.keys()), list(_settings.values()),
                                       n_devices=4, prefix="width64", common_setting=common)

    elif tag == "evaluate":
        common = {"v_shape": [3, 64, 64], "h_dim": 50}
        _settings = {
            "alpha_5e-4_sigma_1e-4": "300000.ckpt.pth",
        }
        run_on_different_setting_evaluate(names=list(_settings.keys()), ckpts=list(_settings.values()),
                                          prefix="width64", n_devices=4, common_setting=common)

