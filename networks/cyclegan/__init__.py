# Acknowledgement:
# The code contained in this file has been generously sourced from the GitHub repository:
#   https://github.com/jhoffman/pytorch-CycleGAN-and-pix2pix
#
# and has been partially revised for this research.

from .cycle_gan_model import CycleGANModel

# from .cycle_gan_model_with_classification_frozen import (
#     CycleGANModelWithClassificationFrozen,
# )
# from .cycle_gan_semantic_model import CycleGANSemanticModel
# from .pix2pix_model import Pix2PixModel
# from .test_model import TestModel


def create_model(opt):
    model = None
    if opt.model == "cycle_gan":
        model = CycleGANModel()
    # elif opt.model == "pix2pix":
    #     assert opt.dataset_mode == "aligned"
    #     model = Pix2PixModel()
    # elif opt.model == "test":
    #     assert opt.dataset_mode == "single"
    #     model = TestModel()
    # elif opt.model == "cycle_gan_with_classification_frozen":
    #     model = CycleGANModelWithClassificationFrozen()
    # elif opt.model == "cycle_gan_semantic":
    #     model = CycleGANSemanticModel()
    else:
        raise NotImplementedError(f"model [{opt.model}] not implemented.")

    model.initialize(opt)
    print(f"model [{model.name()}] was created")
    return model
