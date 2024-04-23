# Acknowledgement:
# The code contained in this file has been generously sourced from the GitHub repository:
#   https://github.com/jhoffman/pytorch-CycleGAN-and-pix2pix
#
# and has been partially revised for this research.

from typing import Literal

from . import networks
from .base_model import BaseModel


class CycleGANModel(BaseModel):
    """
    This is a minimum implementation comparing with the original CycleGAN.
    Only generators are incoporated in this modules.
    Other components, which has nothing to do with inference, has been omitted.
    """

    def name(self):
        return "CycleGANModel"

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        opt = self.opt
        self.model_names = ["G_A", "G_B"]
        self.netG_A = networks.define_G(
            opt.input_nc,
            opt.output_nc,
            opt.ngf,
            opt.which_model_netG,
            opt.norm,
            not opt.no_dropout,
            opt.init_type,
            self.gpu_ids,
        )
        self.netG_B = networks.define_G(
            opt.output_nc,
            opt.input_nc,
            opt.ngf,
            opt.which_model_netG,
            opt.norm,
            not opt.no_dropout,
            opt.init_type,
            self.gpu_ids,
        )

    def set_input(self, input):
        AtoB = self.opt.which_direction == "AtoB"
        self.real_A = input["A" if AtoB else "B"].to(self.device)
        self.real_B = input["B" if AtoB else "A"].to(self.device)
        self.image_paths = input["A_paths" if AtoB else "B_paths"]

    def forward(self):
        self.fake_B = self.netG_A(self.real_A)
        self.rec_A = self.netG_B(self.fake_B)
        self.fake_A = self.netG_B(self.real_B)
        self.rec_B = self.netG_A(self.fake_A)

    def generate_image(self, input, from_domain: Literal["A", "B"]):
        if from_domain == "A":
            return self.netG_A(input.to(self.device))
        elif from_domain == "B":
            return self.netG_B(input.to(self.device))
        else:
            raise ValueError("Invalid domain index.")
