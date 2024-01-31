# Acknowledgement:
# The code contained in this file has been generously sourced from the GitHub repository:
#   https://github.com/jhoffman/pytorch-CycleGAN-and-pix2pix
#
# and has been partially revised for this research.

import itertools

import torch
from pytorch_msssim import MS_SSIM, SSIM
from torchvision.models import VGG16_Weights, vgg16

from lib.cycle_gan.image_pool import ImagePool

from . import networks
from .base_model import BaseModel

activation = {}


def getActivation(name):
    # the hook signature
    def hook(model, input, output):
        activation[name] = output.detach()

    return hook


from torch import nn
from torchvision.models import VGG16_Weights, vgg16

vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).to("cuda")
for param in vgg.parameters():
    param.requires_grad = False
vgg_f1 = nn.Sequential(*list(vgg.children())[0][:7])


def concat3(x):
    return torch.cat((x, x, x), dim=1)


class CycleGANModel(BaseModel):
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

        self.visual_names = visual_names_A + visual_names_B
        # specify the models you want to save to the disk. The program will call base_model.save_networks and base_model.load_networks
        if self.isTrain:
            self.model_names = ["G_A", "G_B", "D_A", "D_B"]
        else:  # during test time, only load Gs
            self.model_names = ["G_A", "G_B"]

        # load/define networks
        # The naming conversion is different from those used in the paper
        # Code (paper): G_A (G), G_B (F), D_A (D_Y), D_B (D_X)
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

        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            self.netD_A = networks.define_D(
                opt.output_nc,
                opt.ndf,
                opt.which_model_netD,
                opt.n_layers_D,
                opt.norm,
                use_sigmoid,
                opt.init_type,
                self.gpu_ids,
            )
            self.netD_B = networks.define_D(
                opt.input_nc,
                opt.ndf,
                opt.which_model_netD,
                opt.n_layers_D,
                opt.norm,
                use_sigmoid,
                opt.init_type,
                self.gpu_ids,
            )

        if self.isTrain:
            self.fake_A_pool = ImagePool(opt.pool_size)
            self.fake_B_pool = ImagePool(opt.pool_size)
            # define loss functions
            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan).to(self.device)
            # self.criterionCycle = lambda x, y: MS_SSIM(
            #     win_size=11, win_sigma=1.5, data_range=1, size_average=True, channel=1
            # )(x, y) + torch.nn.L1Loss()(
            #     x, y
            # )
            self.criterionCycle = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()
            self.criterionSSIM = MS_SSIM(win_size=11, win_sigma=1.5, data_range=1, size_average=True, channel=1)
            # initialize optimizers
            self.optimizer_G = torch.optim.Adam(
                itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999)
            )
            self.optimizer_D = torch.optim.Adam(
                itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999)
            )
            self.optimizers = []
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

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

    def backward_D_basic(self, netD, real, fake):
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        # backward
        loss_D.backward()
        return loss_D

    def backward_D_A(self):
        fake_B = self.fake_B_pool.query(self.fake_B)
        self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_B, fake_B)

    def backward_D_B(self):
        fake_A = self.fake_A_pool.query(self.fake_A)
        self.loss_D_B = self.backward_D_basic(self.netD_B, self.real_A, fake_A)

    def backward_G(self):
        lambda_idt = self.opt.lambda_identity
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B
        # Identity loss
        if lambda_idt > 0:
            # G_A should be identity if real_B is fed.
            self.idt_A = self.netG_A(self.real_B)
            self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_B) * lambda_B * lambda_idt
            # G_B should be identity if real_A is fed.
            self.idt_B = self.netG_B(self.real_A)
            self.loss_idt_B = self.criterionIdt(self.idt_B, self.real_A) * lambda_A * lambda_idt
        else:
            self.loss_idt_A = 0
            self.loss_idt_B = 0

        l1 = torch.nn.L1Loss()
        loss_rec = l1(vgg_f1(concat3(self.real_A)), vgg_f1(concat3(self.fake_B))) + l1(
            vgg_f1(concat3(self.real_B)), vgg_f1(concat3(self.fake_A))
        )

        # GAN loss D_A(G_A(A))
        self.loss_G_A = self.criterionGAN(self.netD_A(self.fake_B), True)
        # GAN loss D_B(G_B(B))
        self.loss_G_B = self.criterionGAN(self.netD_B(self.fake_A), True)
        # Forward cycle loss
        self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A) * lambda_A
        # Backward cycle loss
        self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * lambda_B
        # SSIM loss
        self.loss_ssim_A = 1 - self.criterionSSIM(self.real_A, self.fake_B)
        self.loss_ssim_B = 1 - self.criterionSSIM(self.real_B, self.fake_A)
        # combined loss
        self.loss_G = (
            self.loss_G_A
            + self.loss_G_B
            + self.loss_cycle_A
            + self.loss_cycle_B
            + self.loss_idt_A
            + self.loss_idt_B
            + 1 * self.loss_ssim_A
            + 1 * self.loss_ssim_B
            + loss_rec
        )
        self.loss_G.backward()

    def optimize_parameters(self):
        # forward
        self.forward()
        # G_A and G_B
        self.set_requires_grad([self.netD_A, self.netD_B], False)
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()
        # D_A and D_B
        self.set_requires_grad([self.netD_A, self.netD_B], True)
        self.optimizer_D.zero_grad()
        self.backward_D_A()
        self.backward_D_B()
        self.optimizer_D.step()
