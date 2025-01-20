from dataclasses import dataclass, field

import torch
import torch.nn as nn
import copy

import threestudio
from threestudio.systems.base import BaseLift3DSystem
# import threestudio.systems as ts_sys

from threestudio.utils.ops import binary_cross_entropy, dot
from threestudio.utils.typing import *

from threestudio.utils.misc import process_gpt_file, process_gpt_file_zero123, find_center_of_box, \
    plot_bool_tensor,import_all_modules,find_width_of_box, load_zero123_images
import torch.nn.functional as F
from threestudio.utils.misc import cleanup

from threestudio.utils.config import ExperimentConfig, load_config
import importlib
import inspect

import shutil
import os
import importlib.util


from threestudio.systems.dreamfusion import DreamFusion
from threestudio.systems.prolificdreamer import ProlificDreamer
from threestudio.systems.magic123 import Magic123
from threestudio.systems.zero123_simple import Zero123Simple
from threestudio.systems.zero123 import Zero123
from threestudio.systems.magic3d import Magic3D
from threestudio.systems.latentnerf import LatentNeRF
from threestudio.systems.sjc import ScoreJacobianChaining



cfg = load_config("./configs/control/control-inheritance.yaml")
parent_class = threestudio.find(cfg.inherit_from)

# # Get the attributes of the module
# attributes = [getattr(module, attr_name) for attr_name in dir(module)]
# # Filter out everything that isn't a class from that module
# classes = [attr for attr in attributes if inspect.isclass(attr) and attr.__module__ == parent]

# if len(classes) == 0 or classes[0] is None:
#     raise ValueError(f"Parent class {cfg.inherit_from} not found!")

# parent_class = classes[0]



@threestudio.register("control-system")
class ControlWrapper(parent_class):
    
    @dataclass
    class Config(parent_class.Config):
        inherit_from: str = ""
        gpt_file: str = ""
        recon_loss_weight: float = 0.
        sds_loss_weight: float = 1
        fixed_mapping: bool = True
        scene_edit: bool = False

        save_init_density_grid: bool = True

    cfg: Config


        

    def configure(self):
        
        # create geometry, material, background, renderer
        
        super().configure()
        self.cache_rgb = None
        
        if self.cfg.gpt_file is not None and self.cfg.gpt_file != "":
            if parent_class == Zero123:
                self.objects = process_gpt_file_zero123(self.cfg.gpt_file)
            else:
                self.objects = process_gpt_file(self.cfg.gpt_file)
            [self.log(f"object {o['prompt']} {i}:", o['aabb']) for i,o in enumerate(self.objects)]
            # print("objects:", self.objects)
            [ threestudio.info(f"object  {i}:  {o['prompt']} :{ o['aabb']}") for i,o in enumerate(self.objects)]

        assert hasattr(self, "objects") or self.cfg.prompt_processor.prompt != "", "Either file or prompt should be given!"
        # breakpoint()

        if hasattr(self.renderer, "base_renderer"):
            self.renderer_ref = self.renderer.base_renderer
        else:
            self.renderer_ref = self.renderer

        if hasattr(self, "objects"):
            self.renderer_ref.geometry.aabbs = [ob["aabb"].to(self.device) for ob in self.objects]
        else:
            self.renderer_ref.geometry.aabbs = [ ]
        if self.cfg.scene_edit:
            self.renderer_ref.geometry.aabbs += [self.renderer_ref.bbox.view(-1).to(self.device)]
        
        if self.cfg.recon_loss_weight > 0:
            self.model_copy = copy.deepcopy(self)
            self.model_copy.zero_grad()
        # self.model_copy = copy.deepcopy(self)
        # self.model_copy.zero_grad()


    def on_load_checkpoint(self, checkpoint):
        if any("model_copy" in mod for mod in checkpoint['state_dict'].keys()):
            self.model_copy = copy.deepcopy(self)
        for key in self.state_dict().keys():
            if key not in checkpoint["state_dict"].keys():
                checkpoint["state_dict"][key] = self.state_dict()[key]

        ckpt_keys = list(checkpoint["state_dict"].keys())
        for key in ckpt_keys:
            if "renderer." in key:
                checkpoint["state_dict"][key.replace("renderer.", "renderer_ref.")] = checkpoint["state_dict"][key]
                # del checkpoint["state_dict"][key]
            elif key not in self.state_dict().keys():
                del checkpoint["state_dict"][key]
        
        # self.model_copy = copy.deepcopy(self)
        # self.model_copy.zero_grad()

    def forward(self, batch: Dict[str, Any], aabb=None, void_aabb = None) -> Dict[str, Any]:
        if type(self.renderer_ref) ==  threestudio.models.renderers.nerf_volume_renderer.NeRFVolumeRenderer:
            if aabb is not None:
                self.renderer_ref.set_aabb(aabb)
            elif void_aabb is not None:
                # can be optimized void_aabb to bool
                self.renderer_ref.set_aabb(void_aabb)
                if void_aabb is not None:
                    batch["void_aabb"] = void_aabb
            else:
                pass
                # self.renderer_ref.set_aabb(self.renderer_ref.bbox.view(-1))

        render_out = self.renderer(**batch)
        return {
            **render_out,
        }

    def on_fit_start(self) -> None:
        
        if self.cfg.scene_edit or self.cfg.recon_loss_weight > 0:
            self.model_copy = copy.deepcopy(self)
            self.model_copy.zero_grad()

        super().on_fit_start()
        # only used in training
        #     self.initial_weights = {name: param.clone() for name, param in self.model_copy.named_parameters()}
            
            # del self.model_copy.guidance
            # del self.model_copy.prompt_processor

        # if self.cfg.recon_loss_weight > 0:
        #     self.model_copy = copy.deepcopy(self)
        #     self.model_copy.zero_grad()
        #     self.initial_weights = {name: param.clone() for name, param in self.model_copy.named_parameters()}
            
        # if not hasattr(self, "guidance"):
        #     self.guidance = threestudio.find(self.cfg.guidance_type)(self.cfg.guidance)

        if hasattr(self, "prompt_processor"):
            del self.prompt_processor
        
        processor = threestudio.find(self.cfg.prompt_processor_type)

        if self.cfg.save_init_density_grid:
            os.makedirs(self.get_save_path("density"), exist_ok=True)
            torch.save(self.renderer.estimator.binaries, self.get_save_path("density/init_density_grid.pth"))
            plot_bool_tensor(self.renderer.estimator.binaries, self.get_save_path("density/init_density_grid.png"))

        if hasattr(self, "objects"):
            for obj in self.objects:
                if parent_class == Zero123:
                    c_crossattn, c_concat = self.guidance.prepare_embeddings(obj["prompt"])
                    obj["c_crossattn"], obj["c_concat"] = c_crossattn, c_concat
                    # hard code for now
                    obj['rgb'], obj['mask'], obj['depth'], obj['normal'] = load_zero123_images(obj["prompt"], 128, 128, 0)
                else:
                    self.cfg.prompt_processor["prompt"] = obj["prompt"]
                    obj["prompt_processor"] = processor(
                        self.cfg.prompt_processor
                    )
        else:
            self.objects = [{
                "index": 0,
                "prompt": self.cfg.prompt_processor.prompt,
                "prompt_processor": processor(
                    self.cfg.prompt_processor
                ),
                "aabb": self.renderer_ref.bbox.view(-1)

            }]

        self.object_aabbs = torch.stack([ob["aabb"].to(self.device) for ob in self.objects])
        cleanup()

    def training_step(self, batch, batch_idx):
        # breakpoint()
        loss = 0.0
        batch_ray_o_copy = batch["rays_o"].clone()
        if parent_class == Zero123:
            batch_random_ray_o_copy = batch["random_camera"]["rays_o"].clone()
        for obj in self.objects:
            aabb = obj["aabb"].to(self.device)
            
            aabb_center = find_center_of_box(*aabb).to(self.device)
            # prompt_utils = obj["prompt_processor"]()

            batch["rays_o"] = batch_ray_o_copy + aabb_center
            if parent_class == Zero123:
                batch["random_camera"]["rays_o"] = batch_random_ray_o_copy + aabb_center

            # max_width = torch.max(find_width_of_box(*aabb).to(self.device))
            # batch["rays_o"] += (aabb_center-batch["rays_o"][0,batch_ray_o_copy.shape[1] // 2,batch_ray_o_copy.shape[2] // 2]) * (4.0 - max_width) / 16.0

            if type(self.renderer_ref) == threestudio.models.renderers.nerf_volume_renderer.NeRFVolumeRenderer:
                self.renderer_ref.set_aabb(aabb)

            if parent_class == ProlificDreamer:
                self.prompt_utils = obj["prompt_processor"]()
                # self.prompt_processor = obj["prompt_processor"]
            elif parent_class == Zero123:
                batch["random_camera"]["c_crossattn"],batch["random_camera"]["c_concat"] = obj["c_crossattn"], obj["c_concat"]
                batch["rgb"], batch["mask"], batch["ref_depth"], batch["ref_normal"] = obj["rgb"], obj["mask"], obj["depth"], obj["normal"]
            else:
                self.prompt_processor = obj["prompt_processor"]

            loss += super().training_step(batch, batch_idx)["loss"]

        for name, value in self.cfg.loss.items():
            self.log(f"train_params/{name}", self.C(value))
        
        self.log('train/sds loss', loss, prog_bar=True)

        if self.cfg.scene_edit or self.cfg.recon_loss_weight > 0:
            batch["rays_o"] = batch_ray_o_copy
            batch['bg_color'] =  torch.rand((3,)).to(self.device)   
            threestudio.logger.setLevel("ERROR")
            with torch.no_grad():
                cache_out = self.model_copy.forward(batch, void_aabb=self.object_aabbs)
            out_recon = self(batch, void_aabb=self.object_aabbs)
            threestudio.logger.setLevel("WARNING")


            loss_fn = nn.MSELoss()
            rec_loss = torch.abs(cache_out["comp_rgb"].contiguous() - out_recon["comp_rgb"].contiguous())
            nll_loss = torch.sum(rec_loss) / rec_loss.shape[0]
            nll_loss = (nll_loss + loss_fn(cache_out["comp_rgb"], out_recon["comp_rgb"]))


            nerf_loss = F.smooth_l1_loss(out_recon['comp_rgb'][(out_recon['opacity']>0)[...,0]], cache_out['comp_rgb'][(out_recon['opacity']>0)[...,0]])

            loss = (self.cfg.sds_loss_weight * loss) + (self.cfg.recon_loss_weight * nll_loss + nerf_loss)

            self.log('train/recon_loss', nll_loss, prog_bar=True)

        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        if type(self.renderer_ref) == threestudio.models.renderers.nerf_volume_renderer.NeRFVolumeRenderer:
            self.renderer_ref.set_aabb(self.renderer_ref.bbox.view(-1))
        out = self(batch) 
        
        self.save_image_grid(
            # f"it{self.true_global_step}-val/{batch['index'][0]}.png",
            f"it{self.true_global_step}-{batch['index'][0]}.png",
            [
                {
                    "type": "rgb",
                    "img": out["comp_rgb"][0],
                    "kwargs": {"data_format": "HWC"},
                },
            ]
            + (
                [
                    {
                        "type": "rgb",
                        "img": out["comp_normal"][0],
                        "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                    }
                ]
                if "comp_normal" in out
                else []
            )
            + [
                {
                    "type": "grayscale",
                    "img": out["opacity"][0, :, :, 0],
                    "kwargs": {"cmap": None, "data_range": (0, 1)},
                },
            ],
            name="validation_step",
            step=self.true_global_step,
        )

        objs = []
        batch_ray_o_copy = batch["rays_o"].clone()
        for ab in self.objects:
            aabb = ab["aabb"].to(self.device)
            batch["rays_o"] = batch_ray_o_copy + find_center_of_box(*aabb).to(self.device)
            out = self(batch, aabb=aabb)["comp_rgb"][0]
            objs.append(out)

        self.save_image_grid(
            f"it{self.true_global_step}-{batch['index'][0]}-objects.png",
            # f"it{self.true_global_step}-objects/{batch['index'][0]}.png",
            [
                {
                    "type": "rgb",
                    "img": ob,
                    "kwargs": {"data_format": "HWC"},
                } for ob in objs
            ],
            name="validation_step",
            step=self.true_global_step,
        )
        if self.cfg.scene_edit or self.cfg.recon_loss_weight > 0:
            with torch.no_grad():

                threestudio.logger.setLevel("ERROR")
                out = self(batch, void_aabb=self.object_aabbs)  
                if hasattr(self, "model_copy"):  
                    cache_rgb = self.model_copy.forward(batch, void_aabb=self.object_aabbs)["comp_rgb"]
                else:
                    cache_rgb = None
                threestudio.logger.setLevel("WARNING")


            self.save_image_grid(
                f"it{self.true_global_step}-{batch['index'][0]}-recon.png",
                [
                    {
                        "type": "rgb",
                        "img": out["comp_rgb"][0],
                        "kwargs": {"data_format": "HWC"},
                    },
                ]           + 
                    [
                        {
                            "type": "rgb",
                            "img": cache_rgb[0],
                            "kwargs": {"data_format": "HWC"},
                        }
                    ],
                    # if "comp_normal" in out
                    # else []
                        
                name="validation_step",
                step=self.true_global_step,
            )

    def on_validation_epoch_end(self):
        # filestem = f"it{self.true_global_step}-val"
        # self.save_img_sequence(
        #     filestem,
        #     filestem,
        #     "(\d+)\.png",
        #     save_format="mp4",
        #     fps=30,
        #     name="validation_epoch_end",
        #     step=self.true_global_step,
        # )

        # filestem = f"it{self.true_global_step}-objects"
        # self.save_img_sequence(
        #     filestem,
        #     filestem,
        #     "(\d+)\.png",
        #     save_format="mp4",
        #     fps=30,
        #     name="validation_epoch_end",
        #     step=self.true_global_step,
        # )

        # shutil.rmtree(
        #     os.path.join(self.get_save_dir(), f"it{self.true_global_step}-val")
        # )
        # shutil.rmtree(
        #     os.path.join(self.get_save_dir(), f"it{self.true_global_step}-objects")
        # )
        pass



    def test_step(self, batch, batch_idx):
        if type(self.renderer_ref) == threestudio.models.renderers.nerf_volume_renderer.NeRFVolumeRenderer:

            self.renderer_ref.set_aabb(self.renderer_ref.bbox.view(-1))
        if not self.cfg.scene_edit:
            centers =  [find_center_of_box(*ab["aabb"]) for ab in self.objects]
            difference = torch.mean(torch.stack(centers), dim=0)
            batch["rays_o"] = batch["rays_o"] + difference.to(self.device)
        batch['bg_color'] =  torch.ones((3,)).to(self.device)
        out = self(batch)
        self.save_image_grid(
            f"it{self.true_global_step}-test/{batch['index'][0]}.png",
            [
                {
                    "type": "rgb",
                    "img": out["comp_rgb"][0],
                    "kwargs": {"data_format": "HWC"},
                },
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

# @threestudio.register("control-system")
# def ControlWrapper_helper(cfg: ExperimentConfig, resumed: bool = False) -> BaseLift3DSystem:
#     parent_class = threestudio.find(cfg.inherit_from)
#     class ControlWrapper(parent_class):
#         ...
#     return ControlWrapper(cfg, resumed=resumed)