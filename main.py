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
from monai.utils import set_determinism
from torch.utils.data import ConcatDataset

from dann import DANNInitializer
from segmentation import SegmentationInitializer

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
}


def split_train_data(
    dataset: dict,
    modality: str,
    bg_mapping: dict,
    data_config: dict,
):
    _configs = data_config.copy()
    _configs["modality"] = modality
    _data_class = dataset["class"]
    _train_transforms = dataset["train_transforms"]
    _val_transforms = dataset["val_transforms"]
    ## Training set and validation set are generated by spliting the original training set.
    ## Testing set is the validation set from the original data.
    ## ** note: We test the network without masking any annotation of the given organs.
    ##          Thus mask_mapping is assigned None.
    train_dataset = _data_class(stage="train", transform=_train_transforms, mask_mapping=bg_mapping, **_configs)
    val_dataset = _data_class(stage="train", transform=_val_transforms, mask_mapping=bg_mapping, **_configs)
    test_dataset = _data_class(stage="validation", transform=_val_transforms, mask_mapping=None, **_configs)
    # 10% of the original training data is used as validation set.
    train_dataset = train_dataset[: -int(len(train_dataset) * 0.1)]
    val_dataset = val_dataset[-int(len(val_dataset) * 0.1) :]
    return train_dataset, val_dataset, test_dataset


def get_args():
    parser = argparse.ArgumentParser(description="Segementation branch of DANN, using AMOS dataset.")
    parser.add_argument("--dataset", type=str, default="amos", help="amos / simple_amos")
    parser.add_argument("--modality", type=str, default="ct", help="Modality type: ct / mr / ct+mr")
    parser.add_argument("--masked", action="store_true", help="If true, train with annotation-masked data")
    parser.add_argument(
        "--train_data",
        type=str,
        required=True,
        help="Training data: 'all' (train data)) / 'split' (into training & val sets)",
    )
    # Training module hyperparameters
    parser.add_argument("--mode", type=str, default="train", help="Mode: train / test")
    parser.add_argument("--module", type=str, required=True, help="Module: segmentation / dann")
    parser.add_argument(
        "--pretrained", type=str, help="Checkpointing dir for testing or continuing training procedure."
    )
    parser.add_argument("--batch_size", type=int, default="1", help="Batch size for subject input")
    parser.add_argument("--loss", type=str, default="dice2", help="Loss: dice2 (=DiceCE) / tal")
    parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate for training")
    parser.add_argument("--optim", type=str, default="AdamW", help="Optimizer types: Adam / AdamW")
    parser.add_argument("--max_iter", type=int, default=40000, help="Maximum iteration steps for training")
    parser.add_argument("--eval_step", type=int, default=500, help="Per steps to perform validation")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument(
        "--dev",
        action="store_true",
        help=(
            "Developer mode.",
            "Only a small fraction of data are loaded,",
            "the train_dataloader is not shuffled,",
            "and temp checkpoints are saved in the directory 'debug/'",
        ),
    )
    # Efficiency hyperparameters
    parser.add_argument("--cache_rate", type=float, default=0.1, help="Cache rate to cache your dataset into GPUs")
    parser.add_argument("--num_workers", type=int, default=2, help="Number of workers")
    args = parser.parse_args()
    return args


def get_datasets(
    dataset: dict,
    modality: Literal["ct", "mr", "ct+mr"],
    train_data: Literal["all", "split"],
    masked: bool,
    return_modality_dataset: bool,
    **data_config,
):
    assert "root_dir" in data_config.keys()
    assert "cache_rate" in data_config.keys()
    assert "num_workers" in data_config.keys()
    assert "dev" in data_config.keys()

    print("** Dataset =", dataset["name"])
    print("** Modality =", modality)
    print("** Training set =", train_data)

    data_class = dataset["class"]
    bg = dataset["bg"]
    fg = dataset["fg"]

    if train_data == "all":
        if not masked:
            print("** Foreground =", list(range(1, data_class.num_classes)))
            train_dataset = data_class(stage="train", mask_mapping=None, **data_config)
            val_dataset = data_class(stage="validation", mask_mapping=None, **data_config)
            test_dataset = None
        else:
            # Annotation masked
            print("** Annotation masked = True")
            if modality in ["ct", "mr"]:
                print("** Foreground =", fg[modality])
                train_dataset = data_class(stage="train", mask_mapping=bg[modality], **data_config)
                val_dataset = data_class(stage="validation", mask_mapping=bg[modality], **data_config)
                test_dataset = None
            else:
                print("** Foreground =\n  - ct:", fg["ct"], "\n  - mr:", fg["mr"])
                # Read ct and mr data respectively
                data_config["modality"] = "ct"
                ct_train_dataset = data_class(stage="train", mask_mapping=bg["ct"], **data_config)
                ct_val_dataset = data_class(stage="validation", mask_mapping=bg["ct"], **data_config)
                data_config["modality"] = "mr"
                mr_train_dataset = data_class(stage="train", mask_mapping=bg["mr"], **data_config)
                mr_val_dataset = data_class(stage="validation", mask_mapping=bg["mr"], **data_config)
                # Combine ct and mr data into training and validation set
                train_dataset = ConcatDataset([ct_train_dataset, mr_train_dataset])
                val_dataset = ConcatDataset([ct_val_dataset, mr_val_dataset])
                test_dataset = None
    elif train_data == "split":
        if modality in ["ct", "mr"]:
            if masked:
                print("** Annotation masked = True")
                print("** Foreground =", fg[modality])
                bg_mapping = bg[modality]
            else:
                print("** Foreground =", list(range(1, data_class.num_classes)))
                bg_mapping = None
            train_dataset, val_dataset, test_dataset = split_train_data(dataset, modality, bg_mapping, data_config)
        else:
            if masked:
                print("** Annotation masked = True")
                print("** Foreground =\n  - ct:", fg["ct"], "\n  - mr:", fg["mr"])
                ct_bg_mapping, mr_bg_mapping = bg["ct"], bg["mr"]
            else:
                print("** Foreground =", list(range(1, data_class.num_classes)))
                ct_bg_mapping, mr_bg_mapping = None, None

            ct_train_dataset, ct_val_dataset, ct_test_dataset = split_train_data(
                dataset, "ct", ct_bg_mapping, data_config
            )
            mr_train_dataset, mr_val_dataset, mr_test_dataset = split_train_data(
                dataset, "mr", mr_bg_mapping, data_config
            )
            train_dataset = ConcatDataset([ct_train_dataset, mr_train_dataset])
            val_dataset = ConcatDataset([ct_val_dataset, mr_val_dataset])
            test_dataset = ConcatDataset([ct_test_dataset, mr_test_dataset])
    else:
        raise ValueError("Got an invalid input of option --train_data.")

    if return_modality_dataset:
        return (
            (ct_train_dataset, mr_train_dataset),
            (ct_val_dataset, mr_val_dataset),
            (ct_test_dataset, mr_test_dataset),
        )
    else:
        return train_dataset, val_dataset, test_dataset


def main():
    config = configparser.ConfigParser()
    config.read("./config.ini")
    args = get_args()

    ## Parameters
    # Data
    root = config["path"]["root"]
    output = config["path"]["output"]
    debug = config["path"]["debug"]
    dataset = datasets[args.dataset]
    modality = args.modality
    masked = args.masked
    train_data = args.train_data
    mode = args.mode
    # Module
    module_name = args.module
    pretrained = args.pretrained
    batch_size = args.batch_size
    loss = args.loss
    lr = args.lr
    optim = args.optim
    max_iter = args.max_iter
    eval_step = args.eval_step
    # Efficiency
    deterministic = args.deterministic
    dev = args.dev
    cache_rate = args.cache_rate
    num_workers = args.num_workers

    ## Whether train without randomness
    if deterministic:
        set_determinism(seed=0)
        print("** Deterministic = True")

    ## Dataloaders
    train_dataset, val_dataset, test_dataset = get_datasets(
        dataset=dataset,
        modality=modality,
        train_data=train_data,
        masked=masked,
        return_modality_dataset=modules[module_name]["return_modality_dataset"],
        root_dir=root,
        cache_rate=cache_rate,
        num_workers=num_workers,
        dev=dev,
    )
    train_dataloader, val_dataloader, test_dataloader = modules[module_name]["initializer"].init_dataloaders(
        train_dataset, val_dataset, test_dataset, batch_size, dev
    )

    ## Initialize module
    module = modules[module_name]["initializer"].init_module(
        loss, optim, lr, dataset["class"], modality, masked, dataset["fg"], device
    )
    if pretrained:
        print("** Pretrained checkpoint =", pretrained)
        module.load(pretrained)
    module.to(device)

    ## Train or test
    # ** note: temp checkpoints are saved in the "debug" directory
    #          to separate the result of experiments and temporary
    #          checkpoints generated in developer mode.
    checkpoint_dir = output if not dev else debug
    # create subfolder based on time
    checkpoint_dir = Path(checkpoint_dir) / datetime.now().strftime("%Y%m%d-%H%M%S")
    trainer = modules[module_name]["initializer"].init_trainer(
        max_iter=max_iter,
        eval_step=eval_step,
        checkpoint_dir=checkpoint_dir,
        device=device,
    )

    print("** Mode =", mode)
    if mode == "train":
        trainer.train(module, train_dataloader, val_dataloader)
        # Save command-line arguments
        Path(trainer.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        with open(Path(trainer.checkpoint_dir) / "json", "w") as f:
            json.dump(vars(args), f, indent=4)
    if mode == "test" or test_dataloader:
        test_metric = trainer.validation(module, test_dataloader, label="all")
        print("** Test (Final):", test_metric)
    else:
        raise ValueError("Got an invalid input of option --mode.")


if __name__ == "__main__":
    main()
