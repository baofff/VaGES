from profiles.eblvm.celeba import train_bimdsm
from interface.runner import run_train_profile
from interface.utils.exp_templates import run_on_different_settings


def run_on_different_setting_train(names, settings, prefix="vagesmdsm", n_devices=1, common_setting=None, devices=None, time=None):
    run_on_different_settings(run_fn=run_train_profile, profile=train_bimdsm, path="workspace/runner/eblvm/celeba",
                              prefix=prefix, names=names, settings=settings,
                              n_devices=n_devices, common_setting=common_setting,
                              devices=devices, time=time, time_strategy="now")


if __name__ == "__main__":
    tag = "bimdsm_width64"

    if tag == "bimdsm_width64":
        common = {"v_shape": [3, 64, 64], "h_dim": 50}

        _settings = {
            "default": {},
        }
        run_on_different_setting_train(list(_settings.keys()), list(_settings.values()),
                                       n_devices=4, prefix="bimdsm_width64", common_setting=common)
