from interface.runner import run_train_profile, run_evaluate_profile, run_timing_profile
from profiles.grbm.freyface.train import train_bidsm
from profiles.grbm.freyface.evaluate import evaluate
from interface.utils.exp_templates import run_on_different_settings, evaluate_one_ckpt


def run_on_different_setting_train(names, settings, prefix, n_devices=1, common_setting=None, devices=None, time=None):
    run_on_different_settings(run_fn=run_train_profile, profile=train_bidsm, path="workspace/runner/grbm/freyface",
                              prefix=prefix, names=names, settings=settings,
                              n_devices=n_devices, common_setting=common_setting,
                              devices=devices, time=time, time_strategy="now")


def run_on_different_setting_evaluate(names, ckpts, settings=None, prefix="bidsm", n_devices=1, common_setting=None, devices=None, time=None):
    evaluate_one_ckpt(run_fn=run_evaluate_profile, profile=evaluate, path="workspace/runner/grbm/freyface",
                      prefix=prefix, names=names, ckpts=ckpts, n_devices=n_devices,
                      settings=settings, common_setting=common_setting, devices=devices, time=time)


def run_on_different_setting_timing(names, settings, prefix, n_devices=1, common_setting=None, devices=None, time=None):
    run_on_different_settings(run_fn=run_timing_profile, profile=train_bidsm, path="workspace/runner/grbm/freyface",
                              prefix=prefix, names=names, settings=settings,
                              n_devices=n_devices, common_setting=common_setting,
                              devices=devices, time=time, time_strategy="now")


if __name__ == "__main__":
    tag = "default"

    if tag == "default":
        del train_bidsm["evaluator"]
        n_unrolls = [0, 2, 5]
        n_particles = [2, 4, 6, 8, 10]
        n_lower_steps = [1, 2, 3, 4, 5]
        _settings = {}
        for n_unroll in n_unrolls:
            for n_par in n_particles:
                for n_low in n_lower_steps:
                    _key = "N_%d_L_%d_K_%d" % (n_unroll, n_par, n_low)
                    _settings[_key] = {"criterion": {"kwargs": {"n_unroll": n_unroll, "n_particles": n_par, "n_lower_steps": n_low}}}
        run_on_different_setting_train(list(_settings.keys()), list(_settings.values()), prefix="train_bidsm_default")
        run_on_different_setting_evaluate(list(_settings.keys()), ["best.pth"] * len(_settings), prefix="train_bidsm_default")

    elif tag == "timing":
        train_bidsm['training']['n_its'] = 2000
        n_unrolls = [0, 2, 5]
        n_particles = [2, 4, 6, 8, 10]
        n_lower_steps = [1, 2, 3, 4, 5]
        _settings = {}
        for n_unroll in n_unrolls:
            for n_par in n_particles:
                for n_low in n_lower_steps:
                    _key = "N_%d_L_%d_K_%d" % (n_unroll, n_par, n_low)
                    _settings[_key] = {
                        "criterion": {"kwargs": {"n_unroll": n_unroll, "n_particles": n_par, "n_lower_steps": n_low}}}
        run_on_different_setting_timing(list(_settings.keys()), list(_settings.values()), prefix="timing_bidsm_default")
