import os
import math
import numpy as np
from PIL import Image
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, IterableDataset
import torchvision.transforms.functional as TF

import pytorch_lightning as pl

from .datasets.colmap_utils import \
    read_cameras_binary, read_images_binary, read_points3d_binary
# from models.ray_utils import get_ray_directions
# from utils.misc import get_rank
from tqdm import tqdm

import threestudio
from threestudio import register

from threestudio.utils.config import parse_structured
from threestudio.utils.misc import get_rank, cleanup
from threestudio.utils.ops import (
    get_mvp_matrix,
    get_projection_matrix,
    get_ray_directions,
    get_rays,
)
from threestudio.utils.typing import *


def get_center(pts):
    center = pts.mean(0)
    dis = (pts - center[None,:]).norm(p=2, dim=-1)
    mean, std = dis.mean(), dis.std()
    q25, q75 = torch.quantile(dis, 0.25), torch.quantile(dis, 0.75)
    valid = (dis > mean - 1.5 * std) & (dis < mean + 1.5 * std) & (dis > mean - (q75 - q25) * 1.5) & (dis < mean + (q75 - q25) * 1.5)
    center = pts[valid].mean(0)
    return center

def normalize_poses(poses, pts, up_est_method, center_est_method):
    if center_est_method == 'camera':
        # estimation scene center as the average of all camera positions
        center = poses[...,3].mean(0)
    elif center_est_method == 'lookat':
        # estimation scene center as the average of the intersection of selected pairs of camera rays
        cams_ori = poses[...,3]
        cams_dir = poses[:,:3,:3] @ torch.as_tensor([0.,0.,-1.])
        cams_dir = F.normalize(cams_dir, dim=-1)
        A = torch.stack([cams_dir, -cams_dir.roll(1,0)], dim=-1)
        b = -cams_ori + cams_ori.roll(1,0)
        t = torch.linalg.lstsq(A, b).solution
        center = (torch.stack([cams_dir, cams_dir.roll(1,0)], dim=-1) * t[:,None,:] + torch.stack([cams_ori, cams_ori.roll(1,0)], dim=-1)).mean((0,2))
    elif center_est_method == 'point':
        # first estimation scene center as the average of all camera positions
        # later we'll use the center of all points bounded by the cameras as the final scene center
        center = poses[...,3].mean(0)
    else:
        raise NotImplementedError(f'Unknown center estimation method: {center_est_method}')

    if up_est_method == 'ground':
        # estimate up direction as the normal of the estimated ground plane
        # use RANSAC to estimate the ground plane in the point cloud
        import pyransac3d as pyrsc
        ground = pyrsc.Plane()
        plane_eq, inliers = ground.fit(pts.numpy(), thresh=0.01) # TODO: determine thresh based on scene scale
        plane_eq = torch.as_tensor(plane_eq) # A, B, C, D in Ax + By + Cz + D = 0
        z = F.normalize(plane_eq[:3], dim=-1) # plane normal as up direction
        signed_distance = (torch.cat([pts, torch.ones_like(pts[...,0:1])], dim=-1) * plane_eq).sum(-1)
        if signed_distance.mean() < 0:
            z = -z # flip the direction if points lie under the plane
    elif up_est_method == 'camera':
        # estimate up direction as the average of all camera up directions
        z = F.normalize((poses[...,3] - center).mean(0), dim=0)
    else:
        raise NotImplementedError(f'Unknown up estimation method: {up_est_method}')

    # new axis
    y_ = torch.as_tensor([z[1], -z[0], 0.])
    x = F.normalize(y_.cross(z), dim=0)
    y = z.cross(x)

    if center_est_method == 'point':
        # rotation
        Rc = torch.stack([x, y, z], dim=1)
        R = Rc.T
        poses_homo = torch.cat([poses, torch.as_tensor([[[0.,0.,0.,1.]]]).expand(poses.shape[0], -1, -1)], dim=1)
        inv_trans = torch.cat([torch.cat([R, torch.as_tensor([[0.,0.,0.]]).T], dim=1), torch.as_tensor([[0.,0.,0.,1.]])], dim=0)
        poses_norm = (inv_trans @ poses_homo)[:,:3]
        pts = (inv_trans @ torch.cat([pts, torch.ones_like(pts[:,0:1])], dim=-1)[...,None])[:,:3,0]

        # translation and scaling
        poses_min, poses_max = poses_norm[...,3].min(0)[0], poses_norm[...,3].max(0)[0]
        pts_fg = pts[(poses_min[0] < pts[:,0]) & (pts[:,0] < poses_max[0]) & (poses_min[1] < pts[:,1]) & (pts[:,1] < poses_max[1])]
        center = get_center(pts_fg)
        tc = center.reshape(3, 1)
        t = -tc
        poses_homo = torch.cat([poses_norm, torch.as_tensor([[[0.,0.,0.,1.]]]).expand(poses_norm.shape[0], -1, -1)], dim=1)
        inv_trans = torch.cat([torch.cat([torch.eye(3), t], dim=1), torch.as_tensor([[0.,0.,0.,1.]])], dim=0)
        poses_norm = (inv_trans @ poses_homo)[:,:3]
        scale = poses_norm[...,3].norm(p=2, dim=-1).min()
        poses_norm[...,3] /= scale
        pts = (inv_trans @ torch.cat([pts, torch.ones_like(pts[:,0:1])], dim=-1)[...,None])[:,:3,0]
        pts = pts / scale
    else:
        # rotation and translation
        Rc = torch.stack([x, y, z], dim=1)
        tc = center.reshape(3, 1)
        R, t = Rc.T, -Rc.T @ tc
        poses_homo = torch.cat([poses, torch.as_tensor([[[0.,0.,0.,1.]]]).expand(poses.shape[0], -1, -1)], dim=1)
        inv_trans = torch.cat([torch.cat([R, t], dim=1), torch.as_tensor([[0.,0.,0.,1.]])], dim=0)
        poses_norm = (inv_trans @ poses_homo)[:,:3] # (N_images, 4, 4)

        # scaling
        scale = poses_norm[...,3].norm(p=2, dim=-1).min()
        poses_norm[...,3] /= scale

        # apply the transformation to the point cloud
        pts = (inv_trans @ torch.cat([pts, torch.ones_like(pts[:,0:1])], dim=-1)[...,None])[:,:3,0]
        pts = pts / scale

    return poses_norm, pts

def create_spheric_poses(cameras, n_steps=120):
    # center = torch.as_tensor([0.,0.,0.], dtype=cameras.dtype, device=cameras.device)
    center = torch.as_tensor([0.,0.,0.], dtype=cameras.dtype)

    mean_d = (cameras - center[None,:]).norm(p=2, dim=-1).mean()
    mean_h = cameras[:,2].mean()
    r = (mean_d**2 - mean_h**2).sqrt()
    up = torch.as_tensor([0., 0., 1.], dtype=center.dtype)
    # up = torch.as_tensor([0., 0., 1.], dtype=center.dtype, device=center.device)


    all_c2w = []
    for theta in torch.linspace(0, 2 * math.pi, n_steps):
        cam_pos = torch.stack([r * theta.cos(), r * theta.sin(), mean_h])
        l = F.normalize(center - cam_pos, p=2, dim=0)
        s = F.normalize(l.cross(up), p=2, dim=0)
        u = F.normalize(s.cross(l), p=2, dim=0)
        c2w = torch.cat([torch.stack([s, u, -l], dim=1), cam_pos[:,None]], axis=1)
        all_c2w.append(c2w)

    all_c2w = torch.stack(all_c2w, dim=0)
    
    return all_c2w



@dataclass
class ColmapDataModuleConfig:
    root_dir: str = ""
    scene: str = ""

    img_downscale:int = 4 # specify training image size by either img_wh or img_downscale
    up_est_method: str= "ground" # if true, use estimated ground plane normal direction as up direction
    center_est_method:str = "lookat"
    n_test_traj_steps: int =  120
    apply_mask: bool = False
    load_data_on_gpu: bool = False

    train_num_rays: int = -1
    train_split: str = "train"
    val_split: str = "val"
    test_split: str = "test"
    n_test_traj_steps: int =  120

    num_samples_per_ray: int = 2048
    max_train_num_rays: int = 8192
    dynamic_ray_sampling: bool = True


class ColmapDatasetBase():
    # the data only has to be processed once
    initialized = False
    properties = {}

    def setup(self, cfg, split):
        self.cfg: ColmapDataModuleConfig = cfg
        self.split = split
        self.rank = get_rank()

        if not ColmapDatasetBase.initialized:
            camdata = read_cameras_binary(os.path.join(self.cfg.root_dir, 'sparse/0/cameras.bin'))

            H = int(camdata[1].height)
            W = int(camdata[1].width)

            if 'img_wh' in self.cfg:
                w, h = self.cfg.img_wh
                assert round(W / w * h) == H
            elif 'img_downscale' in self.cfg:
                w, h = int(W / self.cfg.img_downscale + 0.5), int(H / self.cfg.img_downscale + 0.5)
            else:
                raise KeyError("Either img_wh or img_downscale should be specified.")

            img_wh = (w, h)
            factor = w / W

            self.w, self.h = w, h
            self.img_wh = (self.w, self.h)
            
            if camdata[1].model == 'SIMPLE_RADIAL':
                fx = fy = camdata[1].params[0] * factor
                cx = camdata[1].params[1] * factor
                cy = camdata[1].params[2] * factor
            elif camdata[1].model in ['PINHOLE', 'OPENCV']:
                fx = camdata[1].params[0] * factor
                fy = camdata[1].params[1] * factor
                cx = camdata[1].params[2] * factor
                cy = camdata[1].params[3] * factor
            else:
                raise ValueError(f"Please parse the intrinsics for camera model {camdata[1].model}!")
            
            self.directions = get_ray_directions(h,w, (fx, fy), (cx, cy)).to(self.rank)

            imdata = read_images_binary(os.path.join(self.cfg.root_dir, 'sparse/0/images.bin'))

            mask_dir = os.path.join(self.cfg.root_dir, 'masks')
            has_mask = os.path.exists(mask_dir) # TODO: support partial masks
            apply_mask = has_mask and self.cfg.apply_mask
            
            self.all_c2w, self.all_images, self.all_fg_masks = [], [], []

            for i, d in tqdm(enumerate(imdata.values()), total=len(imdata)):
                R = d.qvec2rotmat()
                t = d.tvec.reshape(3, 1)
                c2w = torch.from_numpy(np.concatenate([R.T, -R.T@t], axis=1)).float()
                c2w[:,1:3] *= -1. # COLMAP => OpenGL
                self.all_c2w.append(c2w)
                if self.split in ['train', 'val']:
                    if self.cfg.img_downscale in [2, 4, 8]:
                        img_path = os.path.join(self.cfg.root_dir, f"images_{self.cfg.img_downscale}", d.name)
                    else:
                        img_path = os.path.join(self.cfg.root_dir, 'images', d.name)
                    img = Image.open(img_path)
                    img = img.resize(img_wh, Image.BICUBIC)
                    img = TF.to_tensor(img).permute(1, 2, 0)
                    img = img.to(self.rank) if self.cfg.load_data_on_gpu else img.cpu()
                    if has_mask:
                        mask_paths = [os.path.join(mask_dir, d.name), os.path.join(mask_dir, d.name[3:])]
                        mask_paths = list(filter(os.path.exists, mask_paths))
                        assert len(mask_paths) == 1
                        mask = Image.open(mask_paths[0]).convert('L') # (H, W, 1)
                        mask = mask.resize(img_wh, Image.BICUBIC)
                        mask = TF.to_tensor(mask)[0]
                    else:
                        # mask = torch.ones_like(img[...,0], device=img.device)
                        mask = torch.ones_like(img[...,0])

                    # all_fg_masks.append(mask) # (h, w)
                    # all_images.append(img)
                    self.all_fg_masks.append(img[..., -1]) # (h, w)
                    self.all_images.append(img[...,:3])
            
            self.all_c2w = torch.stack(self.all_c2w, dim=0)   

            pts3d = read_points3d_binary(os.path.join(self.cfg.root_dir, 'sparse/0/points3D.bin'))
            pts3d = torch.from_numpy(np.array([pts3d[k].xyz for k in pts3d])).float()
            self.all_c2w, pts3d = normalize_poses(self.all_c2w, pts3d, up_est_method=self.cfg.up_est_method, center_est_method=self.cfg.center_est_method)

            if self.split == 'test':
                self.all_c2w = create_spheric_poses(self.all_c2w[:,:,3], n_steps=self.cfg.n_test_traj_steps)
                self.all_images = torch.zeros((self.cfg.n_test_traj_steps, self.h, self.w, 3), dtype=torch.float32)
                self.all_fg_masks = torch.zeros((self.cfg.n_test_traj_steps, self.h, self.w), dtype=torch.float32)
            else:
                self.all_images, self.all_fg_masks = \
                    torch.stack(self.all_images, dim=0).float(), \
                    torch.stack(self.all_fg_masks, dim=0).float().to(self.rank)
            
        #     ColmapDatasetBase.properties = {
        #         'w': w,
        #         'h': h,
        #         'img_wh': img_wh,
        #         'factor': factor,
        #         'has_mask': has_mask,
        #         'apply_mask': apply_mask,
        #         # 'directions': directions,
        #         'pts3d': pts3d,
        #         # 'all_c2w': all_c2w,
        #         # 'all_images': all_images,
        #         # 'all_fg_masks': all_fg_masks
        #     }

        #     ColmapDatasetBase.initialized = True
        
        # for k, v in ColmapDatasetBase.properties.items():
        #     setattr(self, k, v)

        # if self.split == 'test':
        #     self.all_c2w = create_spheric_poses(self.all_c2w[:,:,3], n_steps=self.cfg.n_test_traj_steps)
        #     self.all_images = torch.zeros((self.cfg.n_test_traj_steps, self.h, self.w, 3), dtype=torch.float32)
        #     self.all_fg_masks = torch.zeros((self.cfg.n_test_traj_steps, self.h, self.w), dtype=torch.float32)
        # else:
        #     self.all_images, self.all_fg_masks = torch.stack(self.all_images, dim=0).float(), torch.stack(self.all_fg_masks, dim=0).float()

        """
        # for debug use
        from models.ray_utils import get_rays
        rays_o, rays_d = get_rays(self.directions.cpu(), self.all_c2w, keepdim=True)
        pts_out = []
        pts_out.append('\n'.join([' '.join([str(p) for p in l]) + ' 1.0 0.0 0.0' for l in rays_o[:,0,0].reshape(-1, 3).tolist()]))

        t_vals = torch.linspace(0, 1, 8)
        z_vals = 0.05 * (1 - t_vals) + 0.5 * t_vals

        ray_pts = (rays_o[:,0,0][..., None, :] + z_vals[..., None] * rays_d[:,0,0][..., None, :])
        pts_out.append('\n'.join([' '.join([str(p) for p in l]) + ' 0.0 1.0 0.0' for l in ray_pts.view(-1, 3).tolist()]))

        ray_pts = (rays_o[:,0,0][..., None, :] + z_vals[..., None] * rays_d[:,self.h-1,0][..., None, :])
        pts_out.append('\n'.join([' '.join([str(p) for p in l]) + ' 0.0 0.0 1.0' for l in ray_pts.view(-1, 3).tolist()]))

        ray_pts = (rays_o[:,0,0][..., None, :] + z_vals[..., None] * rays_d[:,0,self.w-1][..., None, :])
        pts_out.append('\n'.join([' '.join([str(p) for p in l]) + ' 0.0 1.0 1.0' for l in ray_pts.view(-1, 3).tolist()]))

        ray_pts = (rays_o[:,0,0][..., None, :] + z_vals[..., None] * rays_d[:,self.h-1,self.w-1][..., None, :])
        pts_out.append('\n'.join([' '.join([str(p) for p in l]) + ' 1.0 1.0 1.0' for l in ray_pts.view(-1, 3).tolist()]))
        
        open('cameras.txt', 'w').write('\n'.join(pts_out))
        open('scene.txt', 'w').write('\n'.join([' '.join([str(p) for p in l]) + ' 0.0 0.0 0.0' for l in self.pts3d.view(-1, 3).tolist()]))

        exit(1)
        """

        self.all_c2w = self.all_c2w.float().to(self.rank)
        if self.cfg.load_data_on_gpu:
            self.all_images = self.all_images.to(self.rank) 
            self.all_fg_masks = self.all_fg_masks.to(self.rank)
        

class ColmapDataset(Dataset, ColmapDatasetBase):
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
            directions, c2w, keepdim=True
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
            # "depth": depth,
            "mask": mask,
        }

        return batch
    
    def __getitem__(self, index):
        return self.prepare_data(index)


class ColmapIterableDataset(IterableDataset, ColmapDatasetBase):
    def __init__(self, cfg, split):
        self.setup(cfg, split)
        self.idx = 0
        self.image_perm = torch.randperm(len(self.all_images))

        self.train_num_samples = self.cfg.train_num_rays * self.cfg.num_samples_per_ray
        self.train_num_rays = self.cfg.train_num_rays
        self.mem = 0


    def __iter__(self):
        while True:
            yield {}


    def collate(self, batch) -> Dict[str, Any]:
        idx = self.image_perm[self.idx]

        if (
            self.train_num_rays != -1
            and self.train_num_rays < self.h * self.w
        ):
            # _, height, width, _ = rays_o.shape
            x = torch.randint(
                0, self.w, size=(self.train_num_rays,)
            )
            y = torch.randint(
                0, self.h, size=(self.train_num_rays,)
            )
            # prepare batch data here
            c2w = self.all_c2w[idx]
            light_positions = c2w[..., :3, -1]
            directions = self.directions[y, x]
            rays_o, rays_d = get_rays(
                directions, c2w, keepdim=True
            )
            rays_o, rays_d = rays_o[None], rays_d[None]
            rgb = self.all_images[idx,y, x].view(-1, self.all_images.shape[-1]).to(self.rank)[None]
            # depth = self.all_depths[idx][None]
            mask = self.all_fg_masks[idx, y, x].view(-1).to(self.rank)[None]

            if self.cfg.apply_mask:
                rgb = rgb * mask[...,None] + torch.ones((3,), dtype=torch.float32, device=self.rank) * (1 - mask[...,None])        
                
            rays_o = rays_o.unsqueeze(-2)
            rays_d = rays_d.unsqueeze(-2)
            directions = directions.unsqueeze(-2)
            rgb = rgb.unsqueeze(-2)
            mask = mask.unsqueeze(-2)

        # if torch.cuda.memory_allocated() > self.mem:
        #     self.mem = torch.cuda.memory_allocated()
        #     print('mem: ', self.mem/1024/1024, 'MB')    
        #     # breakpoint()
        if self.cfg.dynamic_ray_sampling:
            # train_num_rays = int(self.train_num_rays * (self.train_num_samples / out['num_samples'].sum().item()))        
            train_num_rays = int(self.train_num_rays * 2)        
            self.train_num_rays = min(int(self.train_num_rays * 0.9 + train_num_rays * 0.1), self.cfg.max_train_num_rays)

        batch = {
            "rays_o": rays_o,
            "rays_d": rays_d,
            "mvp_mtx": None,
            "camera_positions": c2w[..., :3, -1],
            "light_positions": light_positions,
            "elevation": None,
            "azimuth": None,
            "camera_distances": None,
            "rgb": rgb,
            # "depth": depth,
            "mask": mask,
        }

        self.idx += 1
        if self.idx == len(self.all_images):
            self.idx = 0
            self.image_perm = torch.randperm(len(self.all_images))

        return batch

    def collate_old(self, batch) -> Dict[str, Any]:
        idx = self.image_perm[self.idx]
        # prepare batch data here
        c2w = self.all_c2w[idx][None]
        light_positions = c2w[..., :3, -1]
        directions = self.directions[None]
        rays_o, rays_d = get_rays(
            directions, c2w, keepdim=True
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
            and self.train_num_rays < self.h * self.w
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

        batch = {
            "rays_o": rays_o,
            "rays_d": rays_d,
            "mvp_mtx": None,
            "camera_positions": c2w[..., :3, -1],
            "light_positions": light_positions,
            "elevation": None,
            "azimuth": None,
            "camera_distances": None,
            "rgb": rgb,
            # "depth": depth,
            "mask": mask,
        }

        self.idx += 1
        if self.idx == len(self.all_images):
            self.idx = 0
            self.image_perm = torch.randperm(len(self.all_images))

        return batch

@register('colmap-datamodule')
class ColmapDataModule(pl.LightningDataModule):
    def __init__(self, cfg: Optional[Union[dict, DictConfig]] = None) -> None:
        super().__init__()
        self.cfg = parse_structured(ColmapDataModuleConfig, cfg)
    
    def setup(self, stage=None):
        if stage in [None, 'fit']:
            self.train_dataset = ColmapIterableDataset(self.cfg,  self.cfg.train_split)
        if stage in [None, 'fit', 'validate']:
            self.val_dataset = ColmapDataset(self.cfg, self.cfg.val_split)
        if stage in [None, 'test']:
            self.test_dataset = ColmapDataset(self.cfg,  self.cfg.test_split)
        if stage in [None, 'predict']:
            self.predict_dataset = ColmapDataset(self.cfg, self.cfg.train_split)

    def prepare_data(self):
        pass
    
    def general_loader(self, dataset,  batch_size,collate_fn=None) -> DataLoader:
        sampler = None
        return DataLoader(
            dataset, 
            num_workers=0,
            batch_size=batch_size,
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
