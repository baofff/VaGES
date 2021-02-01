# Variational (Gradient) Estimate of the Score Function in Energy-based Latent Variable Models

Code for the paper Variational (Gradient) Estimate of the Score Function in Energy-based Latent Variable Models.


## Requirements

See environment.yml. You can create the environment by running
```
conda env create -f environment.yml
```

## Run VaGES-KSD
To compare VaGES-KSD with KSD and IS-KSD on the toy (checkerboard) dataset, run
```
python run_grbm_toy_ksd.py
```


## Run VaGES-SM
To compare VaGES-SM with baselines on the toy (checkerboard) dataset, run
```
python run_grbm_toy.py
```

To compare VaGES-SM with BiSM on the Frey face dataset, run
```
python run_vagesdsm_grbm_freyface.py
python run_bidsm_grbm_freyface.py
```

To train a deep EBLVM on the cifar10 dataset, run
```
python run_vagesmdsm_eblvm_cifar10.py
```

To train a deep EBLVM on the celeba dataset, run
```
python run_vagesmdsm_eblvm_celeba.py
```


## Run VaGES-Fisher
To estimate the Fisher divergence in GRBMs, run
```
python run_grbm_fisher.py
```


## Remark
* The code will detect free GPUs and run on these GPUs.
You can manually assign GPUs by modify the **devices** argument in functions in the above .py files.

* The downloaded dataset and running result will be saved to **workspace** directory by default.


## Pre-trained models
Pre-trained models on cifar10 and celeba:  [link](https://drive.google.com/file/d/1_lu3xe_pP0lCAnjXsjUMMNfx06I310dC/view?usp=sharing)
