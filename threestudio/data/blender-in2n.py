import os
import json
import math
import numpy as np
from PIL import Image

from dataclasses import dataclass, field

import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset
import torchvision.transforms.functional as TF

import pytorch_lightning as pl

import threestudio
from threestudio import register

# from threestudio.models.ray_utils import get_ray_directions
from threestudio.utils.misc import get_rank

from threestudio.utils.config import parse_structured
from threestudio.utils.misc import get_rank
from threestudio.utils.ops import (
    get_mvp_matrix,
    get_projection_matrix,
    get_ray_directions,
    get_rays,
)
from threestudio.utils.typing import *


@dataclass
class BlenderDataModuleConfig:
    root_dir: str = ""
    scene: str = ""
    batch_size: int = 1
    height: int = 800
    width: int = 800
    load_preprocessed: bool = False
    cam_scale_factor: float = 0.95
    max_num_frames: int = 300
    apply_mask: bool = True
    train_num_rays: int = -1
    train_split: str = "train"
    val_split: str = "val"
    test_split: str = "test"
    img_wh: Tuple[int, int] = (800, 800)
    near_plane: float = 2.0
    far_plane: float = 6.0
    use_random_camera: bool = True
    rays_noise_scale: float = 0.0

    num_samples_per_ray: int = 1024
    max_train_num_rays: int = 8192
    dynamic_ray_sampling: bool = True

class BlenderDatasetBase():
    def setup(self, cfg, split):
        self.split = split
        self.rank = get_rank()
        self.cfg: BlenderDataModuleConfig = cfg

        self.has_mask = True
        self.apply_mask = True

        with open(os.path.join(self.cfg.root_dir, f"transforms_{self.split}.json"), 'r') as f:
            meta = json.load(f)

        if 'w' in meta and 'h' in meta:
            W, H = int(meta['w']), int(meta['h'])
        else:
            W, H = 800, 800

        if 'img_wh' in self.cfg:
            w, h = self.cfg.img_wh
            assert round(W / w * h) == H
        elif 'img_downscale' in self.cfg:
            w, h = W // self.cfg.img_downscale, H // self.cfg.img_downscale
        else:
            raise KeyError("Either img_wh or img_downscale should be specified.")
        
        self.w, self.h = w, h
        self.img_wh = (self.w, self.h)

        self.near, self.far = self.cfg.near_plane, self.cfg.far_plane

        self.focal = 0.5 * w / math.tan(0.5 * meta['camera_angle_x']) # scaled focal length

        # ray directions for all pixels, same for all images (same H, W, focal)
        self.directions = \
            get_ray_directions(self.w, self.h, (self.focal, self.focal), (self.w//2, self.h//2)).to(self.rank) # (h, w, 3)           

        self.all_c2w, self.all_images, self.all_fg_masks = [], [], []
        self.n_frames = len(meta['frames'])
        for i, frame in enumerate(meta['frames']):
            c2w = torch.from_numpy(np.array(frame['transform_matrix'])[:3, :4])
            self.all_c2w.append(c2w)

            img_path = os.path.join(self.cfg.root_dir, f"{frame['file_path']}.png")
            img = Image.open(img_path)
            img = img.resize(self.img_wh, Image.BICUBIC)
            img = TF.to_tensor(img).permute(1, 2, 0) # (4, h, w) => (h, w, 4)

            self.all_fg_masks.append(img[..., -1]) # (h, w)
            self.all_images.append(img[...,:3])

        self.all_c2w, self.all_images, self.all_fg_masks = \
            torch.stack(self.all_c2w, dim=0).float().to(self.rank), \
            torch.stack(self.all_images, dim=0).float().to(self.rank), \
            torch.stack(self.all_fg_masks, dim=0).float().to(self.rank)
        

    def get_all_images(self):
        return self.all_images
    
class BlenderDataset(Dataset, BlenderDatasetBase):
    def __init__(self, cfg, split):
        self.setup(cfg, split)

    def __len__(self):
        return len(self.all_images)
    
    def prepare_data(self, index):

        c2w = self.all_c2w[index]
        # if self.dataset.directions.ndim == 3: # (H, W, 3)
        #     directions = self.dataset.directions
        # elif self.dataset.directions.ndim == 4: # (N, H, W, 3)
        #     directions = self.dataset.directions[index][0]

        light_positions = c2w[..., :3, -1]
        directions = self.directions
        rays_o, rays_d = get_rays(
            directions, c2w, keepdim=True, noise_scale=self.cfg.rays_noise_scale
        )
        # rays_o, rays_d = get_rays(directions, c2w)

        rgb = self.all_images[index]
        # depth = self.all_depths[index]
        mask = self.all_fg_masks[index]

        # TODO: get projection matrix and mvp matrix
        # proj_mtx = get_projection_matrix()

        batch = {
            "index": index,
            "rays_o": rays_o,
            "rays_d": rays_d,
            "mvp_mtx": 0,
            "camera_positions": c2w[..., :3, -1],
            "light_positions": light_positions,
            "elevation": 0,
            "azimuth": 0,
            "camera_distances": 0,
            "rgb": rgb,            
            "c2w": c2w,
            # "depth": depth,
            "mask": mask,
            "height": self.h,
            "width": self.w,
        }

        # c2w = self.all_c2w[index]
        # return {
        #     'index': index,
        #     'c2w': c2w,
        #     'light_positions': c2w[:3, -1],
        #     'H': self.h,
        #     'W': self.w
        # }

        return batch
    

    def __getitem__(self, index):
        return self.prepare_data(index)


class BlenderIterableDataset(IterableDataset, BlenderDatasetBase):
    def __init__(self, cfg, split):
        self.setup(cfg, split)
        self.idx = 0
        self.image_perm = torch.randperm(len(self.all_images))

        self.train_num_samples = self.cfg.train_num_rays * self.cfg.num_samples_per_ray
        self.train_num_rays = self.cfg.train_num_rays
    
    def __iter__(self):
        while True:
            yield {}

    def collate(self, batch) -> Dict[str, Any]:
        idx = self.image_perm[self.idx]
        # prepare batch data here
        c2w = self.all_c2w[idx][None]
        light_positions = c2w[..., :3, -1]
        directions = self.directions[None]
        rays_o, rays_d = get_rays(
            directions, c2w, keepdim=True, noise_scale=self.cfg.rays_noise_scale
        )
        rgb = self.all_images[idx][None]
        # depth = self.all_depths[idx][None]
        mask = self.all_fg_masks[idx][None]

        if self.cfg.dynamic_ray_sampling:
            # train_num_rays = int(self.train_num_rays * (self.train_num_samples / out['num_samples'].sum().item()))        
            train_num_rays = int(self.train_num_rays * 2)        
            self.train_num_rays = min(int(self.train_num_rays * 0.9 + train_num_rays * 0.1), self.cfg.max_train_num_rays)

        if (
            self.train_num_rays != -1
            and self.train_num_rays < self.cfg.height * self.cfg.width
        ):
            _, height, width, _ = rays_o.shape
            x = torch.randint(
                0, width, size=(self.train_num_rays,), device=rays_o.device
            )
            y = torch.randint(
                0, height, size=(self.train_num_rays,), device=rays_o.device
            )

            if self.cfg.apply_mask:
                rgb = rgb * mask[...,None] + torch.ones((3,), dtype=torch.float32, device=self.rank) * (1 - mask[...,None])        
                
            rays_o = rays_o[:, y, x].unsqueeze(-2)
            rays_d = rays_d[:, y, x].unsqueeze(-2)
            directions = directions[:, y, x].unsqueeze(-2)
            rgb = rgb[:, y, x].unsqueeze(-2)
            mask = mask[:, y, x].unsqueeze(-2)
        # if self.cfg.background_color == 'white':
        #     self.model.background_color = torch.ones((3,), dtype=torch.float32, device=self.rank)
        # elif self.cfg.background_color == 'random':
        #     self.model.background_color = torch.rand((3,), dtype=torch.float32, device=self.rank)
        

        
            # depth = depth[:, y, x].unsqueeze(-2)

        # TODO: get projection matrix and mvp matrix
        # proj_mtx = get_projection_matrix()
        # breakpoint()
        index = torch.randint(0, self.n_frames, (1,)).item()
        batch = {
            "index": index,
            "rays_o": rays_o,
            "rays_d": rays_d,
            "mvp_mtx": None,
            "c2w": c2w,
            "camera_positions": c2w[..., :3, -1],
            "light_positions": light_positions,
            "elevation": None,
            "azimuth": None,
            "camera_distances": None,
            "rgb": rgb,
            # "depth": depth,
            "mask": mask,
            "height": self.h,
            "width": self.w,
        }

        # prepare batch data in system
        # c2w = self.all_c2w[idx][None]

        # batch = {
        #     'index': torch.tensor([idx]),
        #     'c2w': c2w,
        #     'light_positions': c2w[..., :3, -1],
        #     'H': self.h,
        #     'W': self.w
        # }

        self.idx += 1
        if self.idx == len(self.all_images):
            self.idx = 0
            self.image_perm = torch.randperm(len(self.all_images))
        # self.idx = (self.idx + 1) % len(self.all_images)

        return batch


@register('blender-datamodule')
class BlenderDataModule(pl.LightningDataModule):
    def __init__(self, cfg: Optional[Union[dict, DictConfig]] = None) -> None:
        super().__init__()
        self.cfg = parse_structured(BlenderDataModuleConfig, cfg)

    def setup(self, stage=None):
        if stage in [None, 'fit']:
            self.train_dataset = BlenderIterableDataset(self.cfg, self.cfg.train_split)
        if stage in [None, 'fit', 'validate']:
            self.val_dataset = BlenderDataset(self.cfg, self.cfg.val_split)
        if stage in [None, 'test']:
            self.test_dataset = BlenderDataset(self.cfg, self.cfg.test_split)
        if stage in [None, 'predict']:
            self.predict_dataset = BlenderDataset(self.cfg, self.cfg.train_split)

    def prepare_data(self):
        pass
    
    def general_loader(self, dataset, batch_size, collate_fn=None) -> DataLoader:
        sampler = None
        return DataLoader(
            dataset,
            num_workers=0,
            batch_size=batch_size,
            # pin_memory=True,
            collate_fn=collate_fn,
        )
    
    def train_dataloader(self):
        return self.general_loader(self.train_dataset, batch_size=1, collate_fn=self.train_dataset.collate)

    def val_dataloader(self):
        return self.general_loader(self.val_dataset, batch_size=1)

    def test_dataloader(self):
        return self.general_loader(self.test_dataset, batch_size=1) 

    def predict_dataloader(self):
        return self.general_loader(self.predict_dataset, batch_size=1)       
