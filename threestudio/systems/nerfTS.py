import os
import random
import shutil

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_efficient_distloss import flatten_eff_distloss

import pytorch_lightning as pl
from pytorch_lightning.utilities.rank_zero import rank_zero_info, rank_zero_debug

import threestudio
# from threestudio.models.ray_utils import get_rays
import threestudio.systems
from threestudio.systems.base import BaseSystem, BaseLift3DSystem
from threestudio.systems.criterions import PSNR
from threestudio.utils.typing import *
from dataclasses import dataclass, field
from threestudio.utils.misc import cleanup, get_device

from threestudio.utils.ops import chunk_batch

# from pytorch_memlab import MemReporter

@threestudio.register('nerf-system-threestudio')
class NeRFSystemThreestudio(BaseLift3DSystem):
    """
    Two ways to print to console:
    1. self.print: correctly handle progress bar
    2. rank_zero_info: use the logging module
    """

    @dataclass
    class Config(BaseLift3DSystem.Config):
        pass

    cfg: Config

    def configure(self):
        # create geometry, material, background, renderer
        super().configure()
        self.criterions = {
            'psnr': PSNR()
        }
        # self.train_num_samples = self.cfg.data.train_num_rays * self.cfg.data.num_samples_per_ray
        # self.train_num_rays = self.cfg.data.train_num_rays

    def forward(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        render_out = self.renderer(**batch)
        return {
            **render_out,
        }
    
    # @profile
    def training_step(self, batch, batch_idx):
        # if self.renderer.cfg.unbounded:
        # torch.cuda.empty_cache()

        # r = torch.load("rays.pth")
        # batch['rays_o'] = r['origin'].view(1,-1,1,3)
        # batch['rays_d'] = r['direction'].view(1,-1,1,3)
        # batch['camera_location'] = r['origin'].view(1,-1,1,3)
        # batch['light_positions'] = r['origin'].view(1,-1,1,3)
        out = self(batch)

        loss = 0.
        # if self.renderer.cfg.unbounded:
        #     loss_rgb = F.smooth_l1_loss(out['comp_rgb'], batch['rgb'])
        # else:
        loss_rgb = F.smooth_l1_loss(out['comp_rgb'][(out['opacity']>0)[...,0]], batch['rgb'][(out['opacity']>0)[...,0]])
        self.log('train/loss_rgb', loss_rgb)
        loss += loss_rgb * self.C(self.cfg.loss.lambda_rgb)

        # distortion loss proposed in MipNeRF360
        # an efficient implementation from https://github.com/sunset1995/torch_efficient_distloss, but still slows down training by ~30%
        if self.C(self.cfg.loss.lambda_distortion) > 0:
            loss_distortion = flatten_eff_distloss(out['weights'], out['points'], out['intervals'], out['ray_indices'])
            self.log('train/loss_distortion', loss_distortion)
            loss += loss_distortion * self.C(self.cfg.loss.lambda_distortion)

        for name, value in self.cfg.loss.items():
            if name.startswith('lambda'):
                self.log(f'train_params/{name}', self.C(value))
        # self.reporter.report(verbose=True)
        self.log('train/loss', loss, prog_bar=True)
        self.log('train/num_rays', batch['rays_o'].shape[1], prog_bar=True)
        return {
            'loss': loss
        }
    
    """
    # aggregate outputs from different devices (DP)
    def training_step_end(self, out):
        pass
    """
    
    """
    # aggregate outputs from different iterations
    def training_epoch_end(self, out):
        pass
    """

    def validation_step(self, batch, batch_idx):
        cleanup()
        out = self(batch)
        # psnr = self.criterions['psnr'](out['comp_rgb'].to(batch['rgb']), batch['rgb'])
        # self.log('val/psnr', psnr, prog_bar=True, rank_zero_only=True)      
        self.save_image_grid(
            f"it{self.true_global_step}-val/{batch['index'][0]}.png",
            (
                [
                    {
                        "type": "rgb",
                        "img": batch["rgb"][0],
                        "kwargs": {"data_format": "HWC"},
                    }
                ]
                if "rgb" in batch
                else []
            )
            + [
                {
                    "type": "rgb",
                    "img": out["comp_rgb"][0],
                    "kwargs": {"data_format": "HWC"},
                },            

                {'type': 'grayscale', 'img': out['depth'][0], 'kwargs': {}},
            ],
            name=f"validation_step_batchidx_{batch_idx}",
            step=self.true_global_step,
        )

    def on_validation_epoch_end(self):
        pass

    def test_step(self, batch, batch_idx):  
        cleanup()
        out = self(batch)
        # breakpoint()
        # psnr = self.criterions['psnr'](out['comp_rgb'].to(batch['rgb']), batch['rgb'])
        if hasattr(self.dataset, 'img_wh'):
            W, H = self.dataset.img_wh
        else:
            # for mutli-vew dataset
            W, H = self.dataset.frame_w, self.dataset.frame_h
        # W, H = self.dataset.img_wh
        self.save_image_grid(f"it{self.true_global_step}-test/{batch['index'][0].item()}.png", [
            {'type': 'rgb', 'img': batch['rgb'].view(H, W, 3), 'kwargs': {'data_format': 'HWC'}},
            {'type': 'rgb', 'img': out['comp_rgb'].view(H, W, 3), 'kwargs': {'data_format': 'HWC'}},
            {'type': 'grayscale', 'img': out['depth'].view(H, W), 'kwargs': {}},
            {'type': 'grayscale', 'img': out['opacity'].view(H, W), 'kwargs': {'cmap': None, 'data_range': (0, 1)}}
        ],            
        name="test_step",
        step=self.true_global_step,
        )

    def on_test_epoch_end(self):
        self.save_img_sequence(
            f"it{self.true_global_step}-test",
            f"it{self.true_global_step}-test",
            "(\d+)\.png",
            save_format="mp4",
            fps=30,
            name="test",
            step=self.true_global_step,
        )