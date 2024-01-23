import datetime
import os
from datetime import datetime

from jsonargparse import CLI, ArgumentParser
from jsonargparse.typing import path_type
from monai.utils import set_determinism
from rich_argparse import RichHelpFormatter
from torch import nn

from lib.datasets.dataset_wrapper import Dataset
from modules.base_trainer import BaseTrainer
from modules.base_updater import BaseUpdater
from modules.base_validator import BaseValidator


def main(
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

    ct_dataloader = ct_data.get_data()
    mr_dataloader = mr_data.get_data()

    trainer.checkpoint_dir += datetime.now().strftime("%Y%m%d-%H%M%S")
    trainer.train(
        module,
        updater,
        ct_dataloader=ct_dataloader,
        mr_dataloader=mr_dataloader,
    )
    performance = evaluator.validation(module, dataloader=(ct_dataloader[2], mr_dataloader[2]))
    print(performance)


if __name__ == "__main__":
    CLI(main, parser_mode="omegaconf")
    # CLI(main, parser_mode="omegaconf", formatter_class=RichHelpFormatter)