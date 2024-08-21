# DeepLake

[![python](https://img.shields.io/badge/Python-3.9.19-3776AB.svg?style=flat&logo=python&logoColor=white)](https://www.python.org)  [![pytorch](https://img.shields.io/badge/PyTorch-2.2.0+cu118-EE4C2C.svg?style=flat&logo=pytorch)](https://pytorch.org) [![Code style: black](https://img.shields.io/badge/Code%20Style-Black-000000.svg)](https://github.com/psf/black)

> Simple deep learning templates. Presented by Laboratory of Analytics on Knowledge Engineering
(LAKE).<br>
> 簡單好用的深度學習訓練模板

This repository is the official implementation of [My Paper Title](https://arxiv.org/abs/2030.12345). 
>📋  Optional: include a graphic explaining your approach/main result, bibtex entry, link to demos, blog posts and tutorials

## Requirements
>📋  Describe how to set up the environment, e.g. pip/conda/docker commands, download datasets, etc...
To install requirements:

```setup
pip install -r requirements.txt
```
## Train and Evaluate Models using CLI

🔥 To **train** the models, run this command:

```train
python train.py --config <path_to_config_file>
```

🔥 To **evaluate** the model, run:

```eval
python test.py --config <path_to_config_file>
```

📋 See the example `[lenet06] train-cli.py` in [notebooks](./notebooks/) to build your own training/testing CLI.


## Imports
This project is setup as a package which means you can now easily import any file into any other file like so:

```python
from deeplake.modules.base.trainer import BaseTrainer
from deeplake.modules.base.updater import BaseUpdater
from deeplake.modules.base.validator import BaseValidator

lenet = LeNet5().cuda()
validator = BaseValidator(metric=batch_acc)
updater = BaseUpdater()
trainer = BaseTrainer(max_iter=1000, eval_step=334, validator=validator)

# train
trainer.train(
  module=lenet,
  updater=updater,
  train_dataloader=train_dataloader,
  val_dataloader=val_dataloader
)

# test
validator.validation(module=lenet, dataloader=test_dataloader)
```

📋 See notebooks [[lenet01-05]](./notebooks/) to use this framework as a package.

## Supports

* [Segmentation Models (smp)](https://github.com/qubvel-org/segmentation_models.pytorch)
* [Monai]()


<!--
## Contributing

>📋  Pick a licence and describe how to contribute to your code repository. 

--> 


