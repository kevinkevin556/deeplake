import itertools
import os
import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from monai.data import DataLoader, decollate_batch
from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric, Metric
from torch import nn, ones, zeros
from torch.nn import BCEWithLogitsLoss
from torch.optim import SGD, Adam, AdamW
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from lib.loss.gan_loss import GANLoss
from lib.utils.image_pool import ImagePool
from networks.cyclegan.base_model import BaseModel
from networks.cyclegan.networks import define_C, define_D, define_G


class CycleGANModel(BaseModel):
    def name(self):
        return "CycleGANModel"

    def initialize(self, opt):
        BaseModel.initialize(self, opt)

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
        self.netG_A = define_G(
            opt.input_nc,
            opt.output_nc,
            opt.ngf,
            opt.which_model_netG,
            opt.norm,
            not opt.no_dropout,
            opt.init_type,
            self.gpu_ids,
        )
        self.netG_B = define_G(
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
            self.netD_A = define_D(
                opt.output_nc,
                opt.ndf,
                opt.which_model_netD,
                opt.n_layers_D,
                opt.norm,
                use_sigmoid,
                opt.init_type,
                self.gpu_ids,
            )
            self.netD_B = define_D(
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
            self.criterionGAN = GANLoss(use_lsgan=not opt.no_lsgan).to(self.device)
            self.criterionCycle = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()
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

        # GAN loss D_A(G_A(A))
        self.loss_G_A = self.criterionGAN(self.netD_A(self.fake_B), True)
        # GAN loss D_B(G_B(B))
        self.loss_G_B = self.criterionGAN(self.netD_B(self.fake_A), True)
        # Forward cycle loss
        self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A) * lambda_A
        # Backward cycle loss
        self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * lambda_B
        # combined loss
        self.loss_G = (
            self.loss_G_A + self.loss_G_B + self.loss_cycle_A + self.loss_cycle_B + self.loss_idt_A + self.loss_idt_B
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


class CycleGANTrainer:
    def __init__(self):
        self.batch_size = opt.batchSize
        self.print_freq = opt.print_freq
        self.display_freq = opt.display_freq
        self.save_latest_freq = opt.save_latest_freq
        self.save_epoch_freq = opt.save_epoch_freq
        self.niter = opt.niter
        self.niter_decay = opt.niter_decay
        self.epoch_count = opt.epoch_count

    def train(self, module, train_dataloader):
        import time

        from data import CreateDataLoader
        from models import create_model
        from options.train_options import TrainOptions
        from util.visualizer import Visualizer

        if __name__ == "__main__":
            opt = TrainOptions().parse()
            data_loader = CreateDataLoader(opt)
            dataset = data_loader.load_data()
            dataset_size = len(data_loader)
            print("#training images = %d" % dataset_size)

            model = create_model(opt)
            model.setup(opt)
            visualizer = Visualizer(opt)
            total_steps = 0

            for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
                epoch_start_time = time.time()
                iter_data_time = time.time()
                epoch_iter = 0

                for i, data in enumerate(dataset):
                    iter_start_time = time.time()
                    if total_steps % opt.print_freq == 0:
                        t_data = iter_start_time - iter_data_time
                    visualizer.reset()
                    total_steps += opt.batchSize
                    epoch_iter += opt.batchSize
                    model.set_input(data)
                    model.optimize_parameters()

                    if total_steps % opt.display_freq == 0:
                        save_result = total_steps % opt.update_html_freq == 0
                        visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

                    if total_steps % opt.print_freq == 0:
                        losses = model.get_current_losses()
                        t = (time.time() - iter_start_time) / opt.batchSize
                        visualizer.print_current_losses(epoch, epoch_iter, losses, t, t_data)
                        if opt.display_id > 0:
                            visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, opt, losses)

                    if total_steps % opt.save_latest_freq == 0:
                        print("saving the latest model (epoch %d, total_steps %d)" % (epoch, total_steps))
                        model.save_networks("latest")

                    iter_data_time = time.time()
                if epoch % opt.save_epoch_freq == 0:
                    print("saving the model at the end of epoch %d, iters %d" % (epoch, total_steps))
                    model.save_networks("latest")
                    model.save_networks(epoch)

                print(
                    "End of epoch %d / %d \t Time Taken: %d sec"
                    % (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time)
                )
                model.update_learning_rate()
