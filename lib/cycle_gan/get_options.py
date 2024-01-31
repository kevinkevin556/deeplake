import dataclasses

import yaml


def get_option(opt_file_path: str):
    start_marker = "------------ Options -------------"
    end_marker = "-------------- End ----------------"

    with open(opt_file_path, "r") as file:
        yaml_text = file.read()

    start_index = yaml_text.find(start_marker)
    end_index = yaml_text.find(end_marker)
    cleaned_yaml_text = yaml_text[start_index + len(start_marker) : end_index].strip()
    opt_dict = yaml.safe_load(cleaned_yaml_text)

    Option = dataclasses.make_dataclass("Option", [(k, type(v)) for k, v in opt_dict.items()])
    opt = Option(**opt_dict)
    return opt
