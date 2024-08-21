# Acknowledgement:
# The code contained in this file has been generously sourced from the GitHub repository:
#   https://github.com/jhoffman/pytorch-CycleGAN-and-pix2pix
#
# and has been partially revised for this research.

from typing import Literal

from torch import nn

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

    def set_input(self, input_data):
        AtoB = self.opt.which_direction == "AtoB"
        self.real_A = input_data["A" if AtoB else "B"].to(self.device)
        self.real_B = input_data["B" if AtoB else "A"].to(self.device)
        self.image_paths = input_data["A_paths" if AtoB else "B_paths"]

    def forward(self):
        self.fake_B = self.netG_A(self.real_A)
        self.rec_A = self.netG_B(self.fake_B)
        self.fake_A = self.netG_B(self.real_B)
        self.rec_B = self.netG_A(self.fake_A)

    def generate_image(self, input_image, from_domain: Literal["A", "B"]):
        if from_domain == "A":
            return self.netG_A(input_image.to(self.device))
        elif from_domain == "B":
            return self.netG_B(input_image.to(self.device))
        else:
            raise ValueError("Invalid domain index.")


class CustomCycleGANModel(BaseModel):
    def name(self):
        return "CycleGANModel"

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        opt = self.opt

        # specify the training losses you want to print out. The program will call base_model.get_current_losses
        self.loss_names = ["D_A", "G_A", "cycle_A", "idt_A", "D_B", "G_B", "cycle_B", "idt_B"]
        # specify the images you want to save/display. The program will call base_model.get_current_visuals
        visual_names_A = ["real_A", "fake_B", "rec_A"]
        visual_names_B = ["real_B", "fake_A", "rec_B"]
        if self.isTrain and self.opt.lambda_identity > 0.0:
            visual_names_A.append("idt_A")
            visual_names_B.append("idt_B")

        # self.visual_names = visual_names_A + visual_names_B
        # specify the models you want to save to the disk. The program will call base_model.save_networks and base_model.load_networks
        if self.isTrain:
            self.model_names = ["G_A", "G_B", "D_A", "D_B"]
        else:  # during test time, only load Gs
            self.model_names = ["G_A", "G_B"]

        # load/define networks
        # The naming conversion is different from those used in the paper
        # Code (paper): G_A (G), G_B (F), D_A (D_Y), D_B (D_X)
        self.netG_A = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.which_model_netG, opt.norm, not opt.no_dropout, opt.init_type, self.gpu_ids)  # fmt: skip
        self.netG_B = networks.define_G(opt.output_nc, opt.input_nc, opt.ngf, opt.which_model_netG, opt.norm, not opt.no_dropout, opt.init_type, self.gpu_ids)  # fmt: skip
        # self.netG_A = nn.Sequential(networks.BasicUNet(spatial_dims=2, in_channels=1, out_channels=1), nn.Tanh()).cuda()
        # self.netG_B = nn.Sequential(networks.BasicUNet(spatial_dims=2, in_channels=1, out_channels=1), nn.Tanh()).cuda()

    def set_input(self, input_data):
        AtoB = self.opt.which_direction == "AtoB"
        self.real_A = input_data["A" if AtoB else "B"].to(self.device)
        self.real_B = input_data["B" if AtoB else "A"].to(self.device)
        self.image_paths = input_data["A_paths" if AtoB else "B_paths"]

    def forward(self):
        self.fake_B = self.netG_A(self.real_A)
        self.rec_A = self.netG_B(self.fake_B)
        self.fake_A = self.netG_B(self.real_B)
        self.rec_B = self.netG_A(self.fake_A)

    def generate_image(self, input_image, from_domain: Literal["A", "B"]):
        if from_domain == "A":
            return self.netG_A(input_image.to(self.device))
        elif from_domain == "B":
            return self.netG_B(input_image.to(self.device))
        else:
            raise ValueError("Invalid domain index.")
