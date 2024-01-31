import datetime
import inspect
import os
import shutil
from datetime import datetime
from pathlib import Path

from jsonargparse import CLI, ArgumentParser
from jsonargparse.typing import Path_fr
from monai.utils import set_determinism
from ruamel.yaml import YAML
from torch import nn

from lib.datasets.dataset_wrapper import Dataset
from modules.base_trainer import BaseTrainer
from modules.base_updater import BaseUpdater
from modules.base_validator import BaseValidator


def setup(
    ct_data: Dataset,
    mr_data: Dataset,
    module: nn.Module,
    updater: BaseUpdater,
    trainer: BaseTrainer,
    evaluator: BaseValidator,
    device: str = "cuda",
    dev: bool = False,
    deterministic: bool = False,
):
    if deterministic:
        set_determinism(seed=0)
    if dev:
        os.environ["MONAI_DEBUG"] = "True"

    suffix = "{time}_{module}_{trainer}_{updater}_LR{lr}_{optimizer}_{ct_dataset}_{mr_dataset}_Step{step}"
    info = {
        "time": datetime.now().strftime("%Y%m%d_%H%M"),
        "module": getattr(module, "alias", module.__class__.__name__),
        "trainer": trainer.get_alias(),
        "updater": updater.get_alias(),
        "lr": module.lr,
        "optimizer": module.optimizer.__class__.__name__,
        "ct_dataset": ct_data.__class__.__name__ if ct_data.in_use else "null",
        "mr_dataset": mr_data.__class__.__name__ if mr_data.in_use else "null",
        "step": trainer.max_iter,
    }
    trainer.checkpoint_dir += suffix.format(**info)
    return ct_data, mr_data, module, trainer, updater, evaluator


def save_config_to(dir_path):
    Path(dir_path).mkdir(exist_ok=True, parents=True)
    target_path = os.path.join(dir_path, "config.yml")

    parser = ArgumentParser()
    parser.add_argument("--config", type=Path_fr)
    cfg_path = parser.parse_args().config

    yaml = YAML()
    yaml.indent(mapping=2, sequence=4, offset=2)
    yaml.preserve_quotes = True
    with open(cfg_path, "r") as stream:
        cfg_data = yaml.load(stream)
    with open(target_path, "w") as file:
        yaml.dump(cfg_data, file)


def save_source_to(dir_path, objects):
    dir_path = Path(dir_path) / "source"
    dir_path.mkdir(exist_ok=True, parents=True)
    source_files = set(Path(inspect.getsourcefile(obj.__class__)) for obj in objects)
    for file in source_files:
        shutil.copy(file, dir_path / file.name)


def main():
    ct_data, mr_data, module, trainer, updater, evaluator = CLI(setup, parser_mode="omegaconf")
    save_config_to(trainer.checkpoint_dir)
    save_source_to(
        trainer.checkpoint_dir,
        objects=[
            ct_data,
            mr_data,
            ct_data.train_transform,
            ct_data.test_transform,
            mr_data.train_transform,
            mr_data.test_transform,
            module,
            trainer,
            updater,
            evaluator,
        ],
    )
    ct_dataloader = ct_data.get_data()
    mr_dataloader = mr_data.get_data()
    trainer.train(
        module,
        updater,
        ct_dataloader=ct_dataloader,
        mr_dataloader=mr_dataloader,
    )
    performance = evaluator.validation(module, dataloader=(ct_dataloader[2], mr_dataloader[2]))
    print(performance)


if __name__ == "__main__":
    main()
    # CLI(main, parser_mode="omegaconf", formatter_class=RichHelpFormatter)
