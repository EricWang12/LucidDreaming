from dataclasses import dataclass, field

import torch
import torch.nn as nn
import copy
import os

import threestudio
from threestudio.systems.base import BaseLift3DSystem
from threestudio.utils.ops import binary_cross_entropy, dot
from threestudio.utils.typing import *

from threestudio.utils.misc import process_gpt_file, find_center_of_box, plot_bool_tensor,find_width_of_box, plot_bool_tensor_new
import torch.nn.functional as F
import warnings

@threestudio.register("dreamfusion-system-edit")
class DreamFusion(BaseLift3DSystem):
    @dataclass
    class Config(BaseLift3DSystem.Config):
        gpt_file: str = ""
        recon_loss_weight: float = 0
        sds_loss_weight: float = 1
        fixed_mapping: bool = True
        scene_edit: bool = False
        save_init_density_grid: bool = True
        global_guidance_start: int = -1

        edit_original: bool = False

    cfg: Config

    def configure(self):
        # create geometry, material, background, renderer
        super().configure()
        self.cache_rgb = None
        
        if self.cfg.gpt_file is not None and self.cfg.gpt_file != "":
            self.objects, self.full_prompt = process_gpt_file(self.cfg.gpt_file, return_full_prompt=True)
            [self.log(f"object {o['prompt']} {i}:", o['aabb']) for i,o in enumerate(self.objects)]
            # print("objects:", self.objects)
            [ threestudio.info(f"object  {i}:  {o['prompt']} :{ o['aabb']}") for i,o in enumerate(self.objects)]

        assert hasattr(self, "objects") or self.cfg.prompt_processor.prompt != "", "Either file or prompt should be given!"
        # breakpoint()
        if hasattr(self, "objects"):
            self.renderer.geometry.aabbs = [ob["aabb"].to(self.device) for ob in self.objects]
        else:
            self.renderer.geometry.aabbs = [ ]
        if self.cfg.scene_edit:
            self.renderer.geometry.aabbs += [self.renderer.bbox.view(-1).to(self.device)]
        
        self.model_copy = copy.deepcopy(self)
        self.model_copy.zero_grad()


    def on_load_checkpoint(self, checkpoint):
        if any("model_copy" in mod for mod in checkpoint['state_dict'].keys()):
            self.model_copy = copy.deepcopy(self)
        for key in self.state_dict().keys():
            if key not in checkpoint["state_dict"].keys():
                checkpoint["state_dict"][key] = self.state_dict()[key]
  
    def forward(self, batch: Dict[str, Any], aabb=None, void_aabb = None) -> Dict[str, Any]:

        if aabb is not None:
            self.renderer.set_aabb(aabb)
        elif void_aabb is not None:
            # can be optimized void_aabb to bool
            self.renderer.set_aabb(void_aabb)
            if void_aabb is not None:
                batch["void_aabb"] = void_aabb
        else:
            self.renderer.set_aabb(self.renderer.bbox.view(-1))

        render_out = self.renderer(**batch)
        return {
            **render_out,
        }

    def on_fit_start(self) -> None:
        super().on_fit_start()
        # only used in training
        if self.cfg.recon_loss_weight > 0:
            self.model_copy = copy.deepcopy(self)
            self.model_copy.zero_grad()
            self.initial_weights = {name: param.clone() for name, param in self.model_copy.named_parameters()}
            
        self.guidance = threestudio.find(self.cfg.guidance_type)(self.cfg.guidance)

        processor = threestudio.find(self.cfg.prompt_processor_type)

        if hasattr(self, "objects"):
            for obj in self.objects:
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
                "aabb": self.renderer.bbox.view(-1)

            }]

        self.object_aabbs = torch.stack([ob["aabb"].to(self.device) for ob in self.objects])
        if self.cfg.global_guidance_start > 0:
            print(f"using global prompt from {self.cfg.global_guidance_start} : {self.full_prompt}")
            self.cfg.prompt_processor["prompt"] = self.full_prompt
            self.full_prompt_util = processor(
                    self.cfg.prompt_processor
                )




    def training_step(self, batch, batch_idx):
        
        loss = 0.0
        batch_ray_o_copy = batch["rays_o"].clone()
        for obj in self.objects:

            prompt_utils = obj["prompt_processor"]()
            if self.cfg.edit_original:
                out = self(batch, aabb=self.renderer.bbox.view(-1))
                # breakpoint()
            else:
                aabb = obj["aabb"].to(self.device)
                
                aabb_center = find_center_of_box(*aabb).to(self.device)

                batch["rays_o"] = batch_ray_o_copy + aabb_center

                max_width = torch.max(find_width_of_box(*aabb).to(self.device))
                batch["rays_o"] += (aabb_center-batch["rays_o"][0,batch_ray_o_copy.shape[1] // 2,batch_ray_o_copy.shape[2] // 2]) * (4.0 - max_width) / 8.0

                out = self(batch, aabb=aabb)
                
            guidance_out = self.guidance(
                out["comp_rgb"], prompt_utils, **batch, rgb_as_latents=False
            )
            for name, value in guidance_out.items():
                self.log(f"train/{name}", value)
                if name.startswith("loss_"):
                    loss += value * self.C(self.cfg.loss[name.replace("loss_", "lambda_")])
                    # loss += loss_weight * value * self.C(self.cfg.loss[name.replace("loss_", "lambda_")])

            if hasattr(self.cfg.loss, "lambda_orient") and self.C(self.cfg.loss.lambda_orient) > 0:
                if "normal" not in out:
                    raise ValueError(
                        "Normal is required for orientation loss, no normal is found in the output."
                    )

                if (out["opacity"] > 0).sum() != 0:
                    
                    loss_orient = (
                        out["weights"].detach()
                        * dot(out["normal"], out["t_dirs"]).clamp_min(0.0) ** 2
                    ).sum() / (out["opacity"] > 0).sum()
                    self.log("train/loss_orient", loss_orient)
                    loss += loss_orient * self.C(self.cfg.loss.lambda_orient)

            loss_sparsity = (out["opacity"] ** 2 + 0.01).sqrt().mean()
            self.log("train/loss_sparsity", loss_sparsity)
            loss += loss_sparsity * self.C(self.cfg.loss.lambda_sparsity)

            opacity_clamped = out["opacity"].clamp(1.0e-3, 1.0 - 1.0e-3)
            loss_opaque = binary_cross_entropy(opacity_clamped, opacity_clamped)
            self.log("train/loss_opaque", loss_opaque)
            loss += loss_opaque * self.C(self.cfg.loss.lambda_opaque)

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
    
        if self.true_global_step == self.cfg.global_guidance_start:
            self.guidance.cfg.max_step_percent=0.3
            self.cfg.recon_loss_weight = 0.1
            self.objects += [
                {
                    "index": len(self.objects),
                    "prompt": self.full_prompt,
                    "prompt_processor": self.full_prompt_util,
                    "aabb": self.renderer.bbox.view(-1)
                }
            ]
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        batch["bg_color"] = torch.ones((3,)).to(self.device)
        out = self(batch) 
        
        self.save_image_grid(
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
            aabb_center = find_center_of_box(*aabb).to(self.device)
            batch["rays_o"] = batch_ray_o_copy + aabb_center

            max_width = torch.max(find_width_of_box(*aabb).to(self.device))
            batch["rays_o"] += (aabb_center-batch["rays_o"][0,batch_ray_o_copy.shape[1] // 2,batch_ray_o_copy.shape[2] // 2]) * (4.0 - max_width) / 8.0


            out = self(batch, aabb=aabb)["comp_rgb"][0]
            objs.append(out)
        self.save_image_grid(
            f"it{self.true_global_step}-{batch['index'][0]}-objects.png",
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

        if self.cfg.save_init_density_grid:
            # plot_bool_tensor_new(self.renderer.estimator.binaries, self.get_save_path(f"density/it{self.true_global_step}-{batch['index'][0]}-density.png"), 2)
            torch.save(self.renderer.estimator.binaries, self.get_save_path(f"density/it{self.true_global_step}-{batch['index'][0]}-density.pth"))
            # breakpoint()
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
                        ] if cache_rgb is not None else [],
                        # if "comp_normal" in out
                        # else []
                            
                    name="validation_step",
                    step=self.true_global_step,
        )


    def on_validation_epoch_end(self):
        pass

    def test_step(self, batch, batch_idx):
        # centers =  [find_center_of_box(*ab["aabb"]) for ab in self.objects]
        # difference = torch.mean(torch.stack(centers), dim=0)
        # batch["rays_o"] = batch["rays_o"] + difference.to(self.device)
        batch["bg_color"] = torch.ones((3,)).to(self.device)
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