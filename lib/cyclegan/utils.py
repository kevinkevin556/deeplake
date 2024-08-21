import dataclasses
from pathlib import Path

import yaml

from networks.cyclegan import CustomCycleGANModel, CycleGANModel


def get_option(opt_file_path: str):
    start_marker = "------------ Options -------------"
    end_marker = "-------------- End ----------------"

    with open(opt_file_path, "r", encoding="utf-8") as file:
        yaml_text = file.read()

    start_index = yaml_text.find(start_marker)
    end_index = yaml_text.find(end_marker)
    cleaned_yaml_text = yaml_text[start_index + len(start_marker) : end_index].strip()
    opt_dict = yaml.safe_load(cleaned_yaml_text)

    Option = dataclasses.make_dataclass("Option", [(k, type(v)) for k, v in opt_dict.items()])
    opt = Option(**opt_dict)
    return opt


def load_cyclegan(checkpoints_dir, option_filename="opt.txt", which_epoch="latest"):
    options = get_option(Path(checkpoints_dir) / option_filename)
    options.isTrain = False
    options.checkpoints_dir = Path(checkpoints_dir).parent.absolute()
    cyclegan = CycleGANModel()
    cyclegan.initialize(options)
    cyclegan.load_networks(which_epoch)
    return cyclegan
