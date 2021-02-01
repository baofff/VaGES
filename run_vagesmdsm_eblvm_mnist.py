from interface.runner import run_train_profile, run_evaluate_profile
from profiles.eblvm.mnist.train_dev import train_vagesmdsm
from profiles.eblvm.mnist.evaluate import evaluate
from interface.utils.exp_templates import run_on_different_settings, evaluate_one_ckpt


def run_on_different_setting_train(names, settings, prefix="vagesmdsm", n_devices=1, common_setting=None, devices=None, time=None):
    run_on_different_settings(run_fn=run_train_profile, profile=train_vagesmdsm, path="workspace/runner/eblvm/mnist",
                              prefix=prefix, names=names, settings=settings,
                              n_devices=n_devices, common_setting=common_setting,
                              devices=devices, time=time, time_strategy="now")


def run_on_different_setting_evaluate(names, ckpts, settings=None, prefix="vagesmdsm", n_devices=1, common_setting=None, devices=None, time=None):
    evaluate_one_ckpt(run_fn=run_evaluate_profile, profile=evaluate, path="workspace/runner/eblvm/mnist",
                      prefix=prefix, names=names, ckpts=ckpts, n_devices=n_devices,
                      settings=settings, common_setting=common_setting, devices=devices, time=time)


if __name__ == "__main__":
    tag = "train"

    if tag == "train":
        _settings = {
            "default": {},
        }
        run_on_different_setting_train(list(_settings.keys()), list(_settings.values()), prefix="vagesmdsm_default")

    elif tag == "evaluate":
        common = {"h_dim": 50}
        _settings = {
            "default": "100000.ckpt.pth",
        }
        run_on_different_setting_evaluate(list(_settings.keys()), list(_settings.values()), prefix="vagesmdsm_default",
                                          common_setting=common)
