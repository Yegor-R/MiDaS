import dataclasses
from typing import List

import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule

from omegaconf import OmegaConf, DictConfig
from src.utils import load_config, print_config
from src.metrics import evaluate_depth_metrics, DepthMetrics
from src.model import MyUnet
from src.sobel import Sobel

class MyAdaptiveMaxPool2d(nn.Module):
    def __init__(self, sz=None):
        super().__init__()


    def forward(self, x):
        inp_size = x.size()
        return nn.functional.max_pool2d(input=x,
                  kernel_size= (inp_size[2], inp_size[3]))

class DepthPLModel(LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.cos = nn.CosineSimilarity(dim=1, eps=0)
        self.get_gradient = Sobel()
        self.model = getattr(smp, config.MODEL.TYPE)(encoder_name=config.MODEL.BACKBONE,
                                                       encoder_weights='imagenet',
                                                       in_channels=3, classes=1,)
        if config.MODEL.TYPE == "PSPNet":
            self.model.segmentation_head = nn.Sequential(
                    nn.Conv2d(512, 1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                    nn.UpsamplingBilinear2d(scale_factor=8.0),
                    nn.Sigmoid()
                    )

        if config.MODEL.TYPE == "Unet":
            self.model.segmentation_head = nn.Sequential(
                                           nn.Conv2d(16,1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                                           nn.Sigmoid()
                                         )

        params = smp.encoders.get_preprocessing_params(config.MODEL.BACKBONE)
        self.register_buffer("std", torch.tensor(params["std"]).view(1, 3, 1, 1))
        self.register_buffer("mean", torch.tensor(params["mean"]).view(1, 3, 1, 1))

    def forward(self, image):
        image = (image - self.mean) / self.std
        output = self.model(image)
        return output

    def training_step(self, batch, batch_idx):
        image, depth = batch['image'], batch['depth']

        # forward
        output = self.model(image)

        # loss: depth
        loss_depth = torch.log(torch.abs(output - depth) + self.config.LOSS.ALPHA).mean()

        # loss: grad
        depth_grad = self.get_gradient(depth)
        output_grad = self.get_gradient(output)
        depth_grad_dx = depth_grad[:, 0, :, :].contiguous().view_as(depth)
        depth_grad_dy = depth_grad[:, 1, :, :].contiguous().view_as(depth)
        output_grad_dx = output_grad[:, 0, :, :].contiguous().view_as(depth)
        output_grad_dy = output_grad[:, 1, :, :].contiguous().view_as(depth)

        loss_dx = torch.log(torch.abs(output_grad_dx - depth_grad_dx) + self.config.LOSS.ALPHA).mean()
        loss_dy = torch.log(torch.abs(output_grad_dy - depth_grad_dy) + self.config.LOSS.ALPHA).mean()

        # loss: normal
        ones = torch.ones(depth.size(0), 1, depth.size(2), depth.size(3), requires_grad=True).type_as(image)
        depth_normal = torch.cat((-depth_grad_dx, -depth_grad_dy, ones), 1)
        output_normal = torch.cat((-output_grad_dx, -output_grad_dy, ones), 1)

        loss_normal = torch.abs(1 - self.cos(output_normal, depth_normal)).mean()

        # loss
        loss = loss_depth \
            + self.config.LOSS.LAMBDA * (loss_dx + loss_dy) \
            + self.config.LOSS.MU * loss_normal

        self.log_dict({'l_d': loss_depth, 'l_g': loss_dx + loss_dy, 'l_n': loss_normal},
                      prog_bar=False, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        image, depth = batch['image'], batch['depth']

        # forward
        output = self.model(image)
        output = torch.nn.functional.interpolate(output, size=[depth.size(2), depth.size(3)],
                                                 mode='bilinear', align_corners=True)

        # calc metrics
        d_metrics: DepthMetrics = evaluate_depth_metrics(output, depth)

        self.log_dict(dataclasses.asdict(d_metrics))

        return d_metrics.mse  # for rmse calculation

    def validation_epoch_end(self, mses: List[float]):
        # todo: val batchsize != 1
        self.log('rmse', np.sqrt(sum(mses) / len(mses)))

    def configure_optimizers(self):
        config = self.config
        optimizer = torch.optim.Adam(self.parameters(),
                                     lr=config.SOLVER.BASE_LR,
                                     weight_decay=config.SOLVER.WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            config.SOLVER.LR_STEP_SIZE,
            config.SOLVER.LR_GAMMA,
        )
        return [optimizer], [scheduler]
