import argparse
import configparser
import json
from datetime import datetime
from pathlib import Path
from typing import Literal

import torch
from medaset.amos import (
    AMOSDataset,
    SimpleAMOSDataset,
    amos_train_transforms,
    amos_val_transforms,
    simple_amos_train_transforms,
    simple_amos_val_transforms,
)
from medaset.lake import SMATDataset, smat_ct_transforms, smat_mr_transforms
from monai.utils import set_determinism
from rich_argparse import RichHelpFormatter
from torch.utils.data import ConcatDataset

from modules.co_training import CoTrainingInitializer
from modules.dann import DANNInitializer
from modules.segmentation import SegmentationInitializer

device = torch.device("cuda")
crop_sample = 2
modules = {
    "segmentation": {
        "initializer": SegmentationInitializer,
        "return_modality_dataset": False,
    },
    "dann": {
        "initializer": DANNInitializer,
        "return_modality_dataset": True,
    },
    "co_training": {
        "initializer": CoTrainingInitializer,
        "return_modality_dataset": True,
    },
}
datasets = {
    "amos": {
        "name": "AMOS",
        "class": AMOSDataset,
        "train_transforms": amos_train_transforms,
        "val_transforms": amos_val_transforms,
        "num_classes": AMOSDataset.num_classes,
        # background and foreground for both modalities (if --masked is applied)
        "fg": {
            "ct": [i for i in range(1, AMOSDataset.num_classes) if i % 2 == 1],
            "mr": [i for i in range(1, AMOSDataset.num_classes) if i % 2 == 0],
        },
        "bg": {
            "ct": {i: 0 for i in range(1, AMOSDataset.num_classes) if i % 2 == 0},
            "mr": {i: 0 for i in range(1, AMOSDataset.num_classes) if i % 2 == 1},
        },
    },
    "simple_amos": {
        "name": "SIMPLE_AMOS",
        "class": SimpleAMOSDataset,
        "train_transforms": simple_amos_train_transforms,
        "val_transforms": simple_amos_val_transforms,
        "num_classes": SimpleAMOSDataset.num_classes,
        # background and foreground for both modalities (if --masked is applied)
        "fg": {
            "ct": [i for i in range(1, SimpleAMOSDataset.num_classes) if i % 2 == 1],
            "mr": [i for i in range(1, SimpleAMOSDataset.num_classes) if i % 2 == 0],
        },
        "bg": {
            "ct": {i: 0 for i in range(1, SimpleAMOSDataset.num_classes) if i % 2 == 0},
            "mr": {i: 0 for i in range(1, SimpleAMOSDataset.num_classes) if i % 2 == 1},
        },
    },
    "smat": {
        "name": "SMAT",
        "class": SMATDataset,
        "train_transforms": None,
        "val_transforms": None,
        "num_classes": SMATDataset.num_classes,
        # 1:SAT, 2:TSM, 3:VAT
        "fg": {
            "ct": [i for i in [1, 2, 3]],
            "mr": [i for i in [1, 2, 3]],
        },
        "bg": {
            "ct": {i: 0 for i in []},
            "mr": {i: 0 for i in []},
        },
    },
}


def split_train_data(data_info: dict, modality: str, bg_mapping: dict, data_config: dict, holdout_ratio: float = 0.1):
    _configs = data_config.copy()
    _configs["modality"] = modality
    _data_class = data_info["class"]
    _train_transforms = data_info["train_transforms"]
    _val_transforms = data_info["val_transforms"]
    ## Training set and validation set are generated by spliting the original training set.
    ## Testing set is the validation set from the original data.
    ## ** note: We test the network without masking any annotation of the given organs.
    ##          Thus mask_mapping is assigned None.
    train_dataset = _data_class(stage="train", transform=_train_transforms, mask_mapping=bg_mapping, **_configs)
    val_dataset = _data_class(stage="train", transform=_val_transforms, mask_mapping=bg_mapping, **_configs)
    test_dataset = _data_class(stage="validation", transform=_val_transforms, mask_mapping=None, **_configs)
    # Default: 10% of the original training data is used as validation set.
    train_dataset = train_dataset[: -int(len(train_dataset) * holdout_ratio)]
    val_dataset = val_dataset[-int(len(val_dataset) * holdout_ratio) :]
    return train_dataset, val_dataset, test_dataset


def get_args():
    parser = argparse.ArgumentParser(description="A CLI for executing modules.", formatter_class=RichHelpFormatter)
    # Data related hyperparameters
    parser.add_argument(
        "--dataset",
        type=str,
        default="amos",
        choices=datasets.keys(),
        help="Specify the dataset you want to use for training or testing.",
    )
    parser.add_argument(
        "--modality",
        type=str,
        default="ct",
        choices=["ct", "mr", "ct+mr", "ct+mr:balance"],
        help="Choose the type of data modality, such as CT scans, MRIs, or a combination.",
    )
    parser.add_argument(
        "--partially_labelled", type=bool, help="If enabled, the training will include partially labeled data."
    )
    parser.add_argument(
        "--holdout_ratio",
        type=float,
        default=0.1,
        help="The proportion of training data allocated for validation (between 0 and 1).",
    )
    # Training module hyperparameters
    parser.add_argument(
        "--mode",
        type=str,
        default="train",
        choices=["train", "test"],
        help="Choose the mode of operation, either `train` or `test`.",
    )
    parser.add_argument(
        "--module", type=str, required=True, choices=modules.keys(), help="Specify the module to execute."
    )
    parser.add_argument(
        "--pretrained",
        type=str,
        help="Provide a deirectory for checkpointing during testing or continuing training procedure.",
    )
    parser.add_argument("--batch_size", type=int, default=1, help="Set the batch size for input data.")
    parser.add_argument(
        "--loss",
        type=str,
        default="dice2",
        choices=["dice2", "tal"],
        help="Choose the loss for training, either `dice2` (DiceCELoss) or `tal`.",
    )
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate for training.")
    parser.add_argument(
        "--optim",
        type=str,
        default="AdamW",
        choices=["Adam", "AdamW", "SGD"],
        help="Select the optimizer for training.",
    )
    parser.add_argument("--max_iter", type=int, default=10000, help="Define the maximum number of training iterations.")
    parser.add_argument(
        "--eval_step", type=int, default=100, help="Set the frequency at which validation is performed during training."
    )
    parser.add_argument("--deterministic", action="store_true", help="Enable deterministic training.")
    # Inference options
    parser.add_argument("--infer_roi_size", type=int, nargs="+", help="Roi size of sliding window inference.")
    parser.add_argument("--infer_batch_size", type=int, help="Batch number of sliding window inference.")
    # Developer mode options
    parser.add_argument(
        "--alpha",
        action="store_true",
        help=(
            "Enable developer mode with reduced data and no shuffling for debugging. "
            "Checkpoints are saved in the directory `debug/`."
        ),
    )
    parser.add_argument(
        "--beta",
        action="store_true",
        help="Enable full dataset loading, training data shuffling, and checkpoint saving in `debug/`.",
    )
    # Efficiency hyperparameters
    parser.add_argument("--cache_rate", type=float, default=0.1, help="Set the rate for caching the dataset into GPUs.")
    parser.add_argument("--num_workers", type=int, default=2, help="Specify the number of data loading workers")
    args = parser.parse_args()
    return args


def get_data_len(dataset):
    if dataset:
        return len(dataset)
    elif dataset is None:
        return 0
    else:
        raise ValueError(f"Expect Dataset or None. Got {dataset}.")


def get_datasets(
    data_info: dict,
    modality: Literal["ct", "mr", "ct+mr"],
    holdout_ratio: float,
    partially_labelled: bool,
    **data_config,
):
    assert "root_dir" in data_config.keys()
    assert "cache_rate" in data_config.keys()
    assert "num_workers" in data_config.keys()
    assert "dev" in data_config.keys()

    print("** Dataset =", data_info["name"])
    print("** Modality =", modality)
    print("** Validation split ratio =", holdout_ratio)

    data_class = data_info["class"]
    background = data_info["bg"]
    foreground = data_info["fg"]

    if partially_labelled:
        print("** Partially Labelled = True")
        if modality in ["ct", "mr"]:
            print("** Foreground =", foreground[modality])
            mapping = {modality: background[modality]}
        else:
            print("** Foreground =\n  - ct:", foreground["ct"], "\n  - mr:", foreground["mr"])
            mapping = {"ct": background["ct"], "mr": background["mr"]}
    else:
        print("** Foreground =", list(range(1, data_class.num_classes)))
        mapping = {}

    if holdout_ratio == 0:
        if modality == "ct":
            ct_train_dataset = data_class(stage="train", mask_mapping=mapping.get("ct", None), **data_config)
            ct_val_dataset = data_class(stage="validation", mask_mapping=mapping.get("ct", None), **data_config)
            ct_test_dataset = None
            mr_train_dataset, mr_val_dataset, mr_test_dataset = None, None, None
        elif modality == "mr":
            ct_train_dataset, ct_val_dataset, ct_test_dataset = None, None, None
            mr_train_dataset = data_class(stage="train", mask_mapping=mapping.get("mr", None), **data_config)
            mr_val_dataset = data_class(stage="validation", mask_mapping=mapping.get("mr", None), **data_config)
            mr_test_dataset = None
        else:
            data_config["modality"] = "ct"
            ct_train_dataset = data_class(stage="train", mask_mapping=mapping.get("ct", None), **data_config)
            ct_val_dataset = data_class(stage="validation", mask_mapping=mapping.get("ct", None), **data_config)
            ct_test_dataset = None
            data_config["modality"] = "mr"
            mr_train_dataset = data_class(stage="train", mask_mapping=mapping.get("mr", None), **data_config)
            mr_val_dataset = data_class(stage="validation", mask_mapping=mapping.get("mr", None), **data_config)
            mr_test_dataset = None

        if modality == "ct+mr:balance":
            train_mul = len(ct_train_dataset) // len(mr_train_dataset)
            val_mul = len(ct_val_dataset) // len(mr_val_dataset)
            mr_train_dataset = ConcatDataset([mr_train_dataset] * train_mul)
            mr_val_dataset = ConcatDataset([mr_val_dataset] * val_mul)

    elif 0 < holdout_ratio < 1:
        if modality == "ct":
            ct_train_dataset, ct_val_dataset, ct_test_dataset = split_train_data(
                data_info, modality, mapping.get("ct", None), data_config
            )
            mr_train_dataset, mr_val_dataset, mr_test_dataset = None, None, None
        elif modality == "mr":
            ct_train_dataset, ct_val_dataset, ct_test_dataset = None, None, None
            mr_train_dataset, mr_val_dataset, mr_test_dataset = split_train_data(
                data_info, modality, mapping.get("mr", None), data_config
            )
        else:
            ct_train_dataset, ct_val_dataset, ct_test_dataset = split_train_data(
                data_info, "ct", mapping.get("ct", None), data_config
            )
            mr_train_dataset, mr_val_dataset, mr_test_dataset = split_train_data(
                data_info, "mr", mapping.get("mr", None), data_config
            )

        if modality == "ct+mr:balance":
            train_mul = len(ct_train_dataset) // len(mr_train_dataset)
            val_mul = len(ct_val_dataset) // len(mr_val_dataset)
            mr_train_dataset = ConcatDataset([mr_train_dataset] * train_mul)
            mr_val_dataset = ConcatDataset([mr_val_dataset] * val_mul)
    else:
        raise ValueError(f"Invalid holdout_ratio. Expect 0 <= holdout_ratio < 1, get {holdout_ratio}.")

    print("** # of Training data =", {"ct": get_data_len(ct_train_dataset), "mr": get_data_len(mr_train_dataset)})
    print("** # of Validation data =", {"ct": get_data_len(ct_val_dataset), "mr": get_data_len(mr_val_dataset)})
    print("** # of Testing data =", {"ct": get_data_len(ct_test_dataset), "mr": get_data_len(mr_test_dataset)})

    return (
        (ct_train_dataset, mr_train_dataset),
        (ct_val_dataset, mr_val_dataset),
        (ct_test_dataset, mr_test_dataset),
    )


def main():
    config = configparser.ConfigParser()
    config.read("./config.ini")
    args = get_args()

    ## Parameters
    # Data
    dataset = args.dataset
    root = config[dataset]["root"]
    output = config["path"]["output"]
    debug = config["path"]["debug"]
    modality = args.modality  # {"ct", "mr", "ct+mr"}
    partially_labelled = args.partially_labelled  # {True, False}
    holdout_ratio = args.holdout_ratio
    mode = args.mode  # {"train", "test"}

    # Module
    module_name = args.module  # {"segmentation", "dann"}
    pretrained = args.pretrained
    batch_size = args.batch_size
    loss = args.loss  # {"dice2", "tal"}
    lr = args.lr
    optim = args.optim
    max_iter = args.max_iter
    eval_step = args.eval_step

    # Development
    deterministic = args.deterministic
    alpha = args.alpha
    beta = args.beta

    # Inference
    infer_roi_size = args.infer_roi_size
    infer_batch_size = args.infer_batch_size

    # Efficiency
    cache_rate = args.cache_rate
    num_workers = args.num_workers

    ## Configurations: data_info and mod_init
    data_info = datasets[dataset]
    mod_init = modules[module_name]["initializer"]

    ## Whether train without randomness
    if deterministic:
        set_determinism(seed=0)
        print("** Deterministic = True")

    ## Dataloaders
    train_dataset, val_dataset, test_dataset = get_datasets(
        data_info=data_info,
        modality=modality,
        holdout_ratio=holdout_ratio,
        partially_labelled=partially_labelled,
        root_dir=root,
        cache_rate=cache_rate,
        num_workers=num_workers,
        dev=alpha,
    )
    train_dataloader, val_dataloader, test_dataloader = mod_init.init_dataloaders(
        train_dataset, val_dataset, test_dataset, batch_size, alpha
    )

    ## Initialize module
    module = mod_init.init_module(
        out_channels=data_info["num_classes"],
        loss=loss,
        optim=optim,
        lr=lr,
        roi_size=infer_roi_size,
        sw_batch_size=infer_batch_size,
        data_info=data_info,
        modality=modality,
        partially_labelled=partially_labelled,
        device=device,
    )
    if pretrained:
        print("** Pretrained checkpoint =", pretrained)
        module.load(pretrained)
    module.to(device)

    ## Train or test
    # ** note: temp checkpoints are saved in the "debug" directory
    #          to separate the result of experiments and temporary
    #          checkpoints generated in developer mode.
    checkpoint_dir = output if not (alpha or beta) else debug
    # create subfolder based on time
    checkpoint_dir = Path(checkpoint_dir) / datetime.now().strftime("%Y%m%d-%H%M%S")
    trainer = mod_init.init_trainer(
        num_classes=data_info["num_classes"],
        max_iter=max_iter,
        eval_step=eval_step,
        checkpoint_dir=checkpoint_dir,
        device=device,
        data_info=data_info,
        partially_labelled=partially_labelled,
    )

    print("** Mode =", mode)
    if mode == "train":
        trainer.train(module, train_dataloader, val_dataloader)
        # Save command-line arguments
        Path(trainer.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        with open(Path(trainer.checkpoint_dir) / "json", "w") as f:
            json.dump(vars(args), f, indent=4)
    if mode == "test" or test_dataloader:
        test_metric = trainer.validation(module, test_dataloader)
        print("** Test (Final):", test_metric)
    else:
        raise ValueError("Got an invalid input of option --mode.")


if __name__ == "__main__":
    main()
