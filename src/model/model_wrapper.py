from dataclasses import dataclass
from pathlib import Path
import gc
from typing import Literal, Optional, Protocol, runtime_checkable

import moviepy.editor as mpy
import torch
import math
import swanlab
from einops import pack, rearrange, repeat
from jaxtyping import Float
from lightning.pytorch import LightningModule
from swanlab.integration.pytorch_lightning import SwanLabLogger
from lightning.pytorch.utilities import rank_zero_only
from tabulate import tabulate
from torch import Tensor, nn
import imageio
import numpy as np

from src.utils.point import get_normal_map

from ..loss.loss_huber import HuberLoss

from ..dataset.data_module import get_data_shim
from ..dataset.types import BatchedExample
from ..evaluation.metrics import compute_lpips, compute_psnr, compute_ssim, abs_relative_difference, delta1_acc
from ..global_cfg import get_cfg
from ..loss import Loss
from ..misc.benchmarker import Benchmarker
from ..misc.cam_utils import rotation_6d_to_matrix
from ..misc.image_io import prep_image, save_image, save_video
from ..misc.LocalLogger import LOG_PATH, LocalLogger
from ..misc.step_tracker import StepTracker
from ..misc.utils import inverse_normalize, vis_depth_map, get_overlap_tag
from ..visualization.annotation import add_label
from ..visualization.camera_trajectory.interpolation import (
    interpolate_extrinsics,
    interpolate_intrinsics,
)
from ..visualization.camera_trajectory.wobble import generate_wobble
from ..visualization.layout import hcat, vcat
from .decoder.decoder import DepthRenderingMode
from .encoder.visualization.encoder_visualizer import EncoderVisualizer

def _normalize(x, eps=1e-8):
    return x / x.norm(dim=-1, keepdim=True).clamp_min(eps)

def _rodrigues_rotate(v, axis, angle):
    # v, axis: [..., 3]; angle: [..., 1]
    axis = _normalize(axis)
    sa, ca = torch.sin(angle), torch.cos(angle)
    v_par  = (v * axis).sum(-1, keepdim=True) * axis
    v_perp = v - v_par
    v_cross = torch.cross(axis, v, dim=-1)
    return v_par + ca * v_perp + sa * v_cross

def _closest_approach_midpoint(C0, d0, C1, d1, eps=1e-6):
    # d0 and d1 are assumed to be unit vectors.
    v, w = d0, d1
    r = C0 - C1
    vw = (v * w).sum(-1, keepdim=True)
    denom = (1.0 - vw * vw).clamp_min(eps)
    s = ( vw * (r * w).sum(-1, keepdim=True) - (r * v).sum(-1, keepdim=True) ) / denom
    t = ( (r * w).sum(-1, keepdim=True) - vw * (r * v).sum(-1, keepdim=True) ) / denom
    P0 = C0 + s * v
    P1 = C1 + t * w
    mid = 0.5 * (P0 + P1)
    parallel = (1.0 - vw.abs()).squeeze(-1) < 1e-3
    return torch.where(parallel[..., None], 0.5 * (C0 + C1), mid)

@torch.no_grad()
def generate_pivot_orbit_face_initial(
    extrinsic0, extrinsic1, t,
    theta_max_deg=5.0,
    axis="X",
    forward_push_mean=0.02,
    forward_push_std=0.02,
    unit_scale=1.0,
    per_frame_noise=False,
):
    device = extrinsic0.device
    dtype  = extrinsic0.dtype
    B, T = extrinsic0.shape[0], t.shape[0]

    R0 = extrinsic0[:, :3, :3]
    C0 = extrinsic0[:, :3, 3]
    R1 = extrinsic1[:, :3, :3]
    C1 = extrinsic1[:, :3, 3]
    d0 = _normalize(R0[:, :, 2])
    d1 = _normalize(R1[:, :, 2])

    pivot = _closest_approach_midpoint(C0, d0, C1, d1)

    v0 = C0 - pivot
    r_axis_world = R0[:, :, 0] if axis.upper() == "X" else R0[:, :, 1]

    theta_max = theta_max_deg * math.pi / 180.0
    t_smooth = (torch.cos(math.pi * (t + 1)) + 1) / 2
    angles = theta_max * torch.sin(2 * math.pi * t_smooth)  # [T]

    if per_frame_noise:
        push = (torch.randn(B, T, 1, device=device, dtype=dtype) * forward_push_std
                + forward_push_mean).clamp_min(0.0) * unit_scale
    else:
        push_val = (torch.randn(B, 1, device=device, dtype=dtype) * forward_push_std
                    + forward_push_mean).clamp_min(0.0) * unit_scale
        push = push_val.unsqueeze(1).expand(B, T, 1)  # [B,T,1]

    E = torch.zeros(B, T, 4, 4, device=device, dtype=dtype)
    E[:, :, 3, 3] = 1.0

    E[:, 0, :3, :3] = R0
    E[:, 0, :3, 3]  = C0

    for k in range(1, T):
        a = angles[k].view(B, 1)
        v_t = _rodrigues_rotate(v0, r_axis_world, a)         # [B,3]
        C_t = pivot + v_t                                    # [B,3]

        ax = r_axis_world
        x0, y0, z0 = R0[:, :, 0], R0[:, :, 1], R0[:, :, 2]
        x_t = _rodrigues_rotate(x0, ax, a)
        y_t = _rodrigues_rotate(y0, ax, a)
        z_t = _rodrigues_rotate(z0, ax, a)
        R_t = torch.stack([x_t, y_t, z_t], dim=-1)           # [B,3,3]

        fwd = R_t[:, :, 2]                                   # [B,3]
        C_t = C_t + push[:, k] * fwd                         # [B,3]

        E[:, k, :3, :3] = R_t
        E[:, k, :3, 3]  = C_t

    return E

@dataclass
class OptimizerCfg:
    lr: float
    warm_up_steps: int
    backbone_lr_multiplier: float


@dataclass
class TestCfg:
    output_path: Path
    align_pose: bool
    pose_align_steps: int
    rot_opt_lr: float
    trans_opt_lr: float
    compute_scores: bool
    save_image: bool
    save_video: bool
    save_compare: bool
    generate_video: bool
    mode: Literal["inference", "evaluation"]
    image_folder: str


@dataclass
class TrainCfg:
    output_path: Path
    depth_mode: DepthRenderingMode | None
    print_log_every_n_steps: int
    distiller: str
    distill_max_steps: int
    pose_loss_alpha: float = 1.0
    pose_loss_delta: float = 1.0
    cxt_depth_weight: float = 0.01
    pm_loss_weight: float = 0.01
    weight_pose: float = 1.0
    weight_depth: float = 1.0
    weight_normal: float = 1.0
    render_ba: bool = False
    render_ba_after_step: int = 0


@runtime_checkable
class TrajectoryFn(Protocol):
    def __call__(
        self,
        t: Float[Tensor, " t"],
    ) -> tuple[
        Float[Tensor, "batch view 4 4"],  # extrinsics
        Float[Tensor, "batch view 3 3"],  # intrinsics
    ]:
        pass


class ModelWrapper(LightningModule):
    logger: Optional[SwanLabLogger]
    model: nn.Module
    losses: nn.ModuleList
    optimizer_cfg: OptimizerCfg
    test_cfg: TestCfg
    train_cfg: TrainCfg
    step_tracker: StepTracker | None

    def __init__(
        self,
        optimizer_cfg: OptimizerCfg,
        test_cfg: TestCfg,
        train_cfg: TrainCfg,
        model: nn.Module,
        losses: list[Loss],
        step_tracker: StepTracker | None
    ) -> None:
        super().__init__()
        self.optimizer_cfg = optimizer_cfg
        self.test_cfg = test_cfg
        self.train_cfg = train_cfg
        self.step_tracker = step_tracker
        
        # Set up the model.
        self.encoder_visualizer = None
        self.model = model
        self.data_shim = get_data_shim(self.model.encoder)
        self.losses = nn.ModuleList(losses)
        
        if self.model.encoder.pred_pose:
            self.loss_pose = HuberLoss(alpha=self.train_cfg.pose_loss_alpha, delta=self.train_cfg.pose_loss_delta)

        # This is used for testing.
        self.benchmarker = Benchmarker()
        
    def compute_pointmap_loss(self, depth_dict):
        stud_pts = depth_dict["pts"]
        distill_infos = depth_dict.get("distill_infos", {})
        teach_pts = distill_infos.get("pts_all", None)
        if teach_pts is None:
            return torch.tensor(0.0, dtype=torch.float32, device=stud_pts.device)

        teach_mask, stud_mask = distill_infos.get("conf_mask", None), depth_dict.get("conf_valid_mask", None) 
        valid_mask = stud_mask & teach_mask 
        if not valid_mask.any():
            return torch.tensor(0.0, dtype=torch.float32, device=stud_pts.device)
        # Compute L2 distance between predicted and ground truth points
        pm_loss = torch.norm(teach_pts[valid_mask] - stud_pts[valid_mask], dim=-1).mean()

        return self.train_cfg.pm_loss_weight * pm_loss

    def on_train_epoch_start(self) -> None:
        # our custom dataset and sampler has to have epoch set by calling set_epoch
        if hasattr(self.trainer.datamodule.train_loader.dataset, "set_epoch"):
            self.trainer.datamodule.train_loader.dataset.set_epoch(self.current_epoch)
        if hasattr(self.trainer.datamodule.train_loader.sampler, "set_epoch"):
            self.trainer.datamodule.train_loader.sampler.set_epoch(self.current_epoch)

    def on_validation_epoch_start(self) -> None:
        print(f"Validation epoch start on rank {self.trainer.global_rank}")
        # our custom dataset and sampler has to have epoch set by calling set_epoch
        if hasattr(self.trainer.datamodule.val_loader.dataset, "set_epoch"):
            self.trainer.datamodule.val_loader.dataset.set_epoch(self.current_epoch)
        if hasattr(self.trainer.datamodule.val_loader.sampler, "set_epoch"):
            self.trainer.datamodule.val_loader.sampler.set_epoch(self.current_epoch)
        
    def training_step(self, batch, batch_idx):
        # combine batch from different dataloaders
        # torch.cuda.empty_cache()
        if isinstance(batch, list):
            batch_combined = None
            for batch_per_dl in batch:
                if batch_combined is None:
                    batch_combined = batch_per_dl
                else:
                    for k in batch_combined.keys():
                        if isinstance(batch_combined[k], list):
                            batch_combined[k] += batch_per_dl[k]
                        elif isinstance(batch_combined[k], dict):
                            for kk in batch_combined[k].keys():
                                batch_combined[k][kk] = torch.cat([batch_combined[k][kk], batch_per_dl[k][kk]], dim=0)
                        else:
                            raise NotImplementedError
            batch = batch_combined

        # Run the model.
        visualization_dump = None
        encoder_output, output = self.model(batch["context"]["image"], self.global_step, visualization_dump=visualization_dump)
        gaussians, pred_pose_enc_list, pred_context_pose = encoder_output.gaussians, encoder_output.pred_pose_enc_list, encoder_output.pred_context_pose
        # gaussians, pred_pose, pred_context_pose = encoder_output.gaussians, encoder_output.pred_pose, encoder_output.pred_context_pose
        depth_dict, infos, distill_infos = encoder_output.depth_dict, encoder_output.infos, encoder_output.distill_infos
        
        num_context_views = pred_context_pose['extrinsic'].shape[1]

        using_index = torch.arange(num_context_views, device=gaussians.means.device)
        batch["using_index"] = using_index
        
        target_gt = batch["context"]["image"]
        # scene_scale = infos["scene_scale"]
        self.log("train/scene_scale", infos["scene_scale"])
        self.log("train/voxelize_ratio", infos["voxelize_ratio"])

        # Compute metrics.
        psnr_probabilistic = compute_psnr(
            rearrange(target_gt, "b v c h w -> (b v) c h w"),
            rearrange(output.color, "b v c h w -> (b v) c h w"),
        )
        self.log("train/psnr_probabilistic", psnr_probabilistic.mean())

        consis_absrel = abs_relative_difference(
            rearrange(output.depth, "b v h w -> (b v) h w"),
            rearrange(depth_dict['depth'].squeeze(-1), "b v h w -> (b v) h w"),
            rearrange(distill_infos['conf_mask'], "b v h w -> (b v) h w"),
        )
        self.log("train/consis_absrel", consis_absrel.mean())

        consis_delta1 = delta1_acc(
            rearrange(output.depth, "b v h w -> (b v) h w"),
            rearrange(depth_dict['depth'].squeeze(-1), "b v h w -> (b v) h w"),
            rearrange(distill_infos['conf_mask'], "b v h w -> (b v) h w"),
        )
        self.log("train/consis_delta1", consis_delta1.mean())
        
        # Compute and log loss.
        total_loss = 0

        depth_dict['distill_infos'] = distill_infos
        with torch.amp.autocast('cuda', enabled=False):
            for loss_fn in self.losses: # RGB_MSE + Lipips + Depth
                loss = loss_fn.forward(output, batch, gaussians, depth_dict, self.global_step)
                self.log(f"loss/{loss_fn.name}", loss)
                total_loss = total_loss + loss

            if depth_dict is not None and "depth" in get_cfg()["loss"].keys() and self.train_cfg.cxt_depth_weight > 0:
                depth_loss_idx = list(get_cfg()["loss"].keys()).index("depth")
                depth_loss_fn = self.losses[depth_loss_idx].ctx_depth_loss
                loss_depth = depth_loss_fn(depth_dict["depth_map"], depth_dict["depth_conf"], batch, cxt_depth_weight=self.train_cfg.cxt_depth_weight)
                self.log("loss/ctx_depth", loss_depth)
                total_loss = total_loss + loss_depth

        self.log("loss/total", total_loss)
        print(f"\033[34m== total_loss: {total_loss} ==\033[0m")

        # Skip batch if loss is too high after certain step
        SKIP_AFTER_STEP = 1000  
        LOSS_THRESHOLD = 0.2
        if self.global_step > SKIP_AFTER_STEP and total_loss > LOSS_THRESHOLD:
            print(f"Skipping batch with high loss ({total_loss:.6f}) at step {self.global_step} on Rank {self.global_rank}")
            # set to a really small number
            return total_loss * 1e-10

        if self.global_rank == 0 and self.global_step % self.train_cfg.print_log_every_n_steps == 0:
            print(
                f"\033[92mtrain step {self.global_step}; \n"
                f"scene = {[x[:20] for x in batch['scene']]}; \n"
                f"context = {batch['context']['index'].tolist()}; \n"
                f"loss = {total_loss:.6f}; \033[0m"
            )
            
        self.log("info/global_step", self.global_step)  # hack for ckpt monitor
        
        # Tell the data loader processes about the current step.
        if self.step_tracker is not None:
            self.step_tracker.set_step(self.global_step)
        
        del batch
        if self.global_step % 50 == 0:
            gc.collect()
            torch.cuda.empty_cache()

        return total_loss
    
    def on_after_backward(self):
        total_norm = 0.0
        counter = 0
        for p in self.parameters():
            if p.grad is not None:
                param_norm = p.grad.detach().data.norm(2)
                total_norm += param_norm.item() ** 2
                counter += 1
        total_norm = (total_norm / counter) ** 0.5
        self.log("loss/grad_norm", total_norm)
        
    def test_step(self, batch, batch_idx):
        batch: BatchedExample = self.data_shim(batch)
        b, v, _, h, w = batch["target"]["image"].shape
        assert b == 1
        if batch_idx % 100 == 0:
            print(f"Test step {batch_idx:0>6}.")
        
        # Render Gaussians.
        with self.benchmarker.time("encoder"):
            gaussians = self.model.encoder(
                (batch["context"]["image"]+1)/2,
                self.global_step,
            )[0]
        # export_ply(gaussians.means[0], gaussians.scales[0], gaussians.rotations[0], gaussians.harmonics[0], gaussians.opacities[0], Path("gaussians.ply"))
        # align the target pose
        if self.test_cfg.align_pose:
            output = self.test_step_align(batch, gaussians)
        else:
            with self.benchmarker.time("decoder", num_calls=v):
                output = self.model.decoder.forward(
                    gaussians,
                    batch["target"]["extrinsics"],
                    batch["target"]["intrinsics"],
                    batch["target"]["near"],
                    batch["target"]["far"],
                    (h, w),
                )
        
        # compute scores
        if self.test_cfg.compute_scores:
            overlap = batch["context"]["overlap"][0]
            overlap_tag = get_overlap_tag(overlap)

            rgb_pred = output.color[0]
            rgb_gt = batch["target"]["image"][0]
            all_metrics = {
                f"lpips_ours": compute_lpips(rgb_gt, rgb_pred).mean(),
                f"ssim_ours": compute_ssim(rgb_gt, rgb_pred).mean(),
                f"psnr_ours": compute_psnr(rgb_gt, rgb_pred).mean(),
            }
            methods = ['ours']

            self.log_dict(all_metrics)
            self.print_preview_metrics(all_metrics, methods, overlap_tag=overlap_tag)
        
        # Save images.
        (scene,) = batch["scene"]
        name = get_cfg()["swanlab"]["name"]
        path = self.test_cfg.output_path / name
        if self.test_cfg.save_image:
            for index, color in zip(batch["target"]["index"][0], output.color[0]):
                save_image(color, path / scene / f"color/{index:0>6}.png")

        if self.test_cfg.save_video:
            frame_str = "_".join([str(x.item()) for x in batch["context"]["index"][0]])
            save_video(
                [a for a in output.color[0]],
                path / "video" / f"{scene}_frame_{frame_str}.mp4",
            )

        if self.test_cfg.save_compare:
            # Construct comparison image.
            context_img = inverse_normalize(batch["context"]["image"][0])
            comparison = hcat(
                add_label(vcat(*context_img), "Context"),
                add_label(vcat(*rgb_gt), "Target (Ground Truth)"),
                add_label(vcat(*rgb_pred), "Target (Prediction)"),
            )
            save_image(comparison, path / f"{scene}.png")
                
    def test_step_align(self, batch, gaussians):
        self.model.encoder.eval()
        # freeze all parameters
        for param in self.model.encoder.parameters():
            param.requires_grad = False

        b, v, _, h, w = batch["target"]["image"].shape
        output_c2ws = batch["target"]["extrinsics"]
        with torch.set_grad_enabled(True):
            cam_rot_delta = nn.Parameter(torch.zeros([b, v, 6], requires_grad=True, device=output_c2ws.device))
            cam_trans_delta = nn.Parameter(torch.zeros([b, v, 3], requires_grad=True, device=output_c2ws.device))
            opt_params = []
            self.register_buffer("identity", torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]).to(output_c2ws))
            opt_params.append(
                {
                    "params": [cam_rot_delta],
                    "lr": 0.005,
                }
            )
            opt_params.append(
                {
                    "params": [cam_trans_delta],
                    "lr": 0.005,
                }
            )
            pose_optimizer = torch.optim.Adam(opt_params)
            extrinsics = output_c2ws.clone()
            with self.benchmarker.time("optimize"):
                for i in range(self.test_cfg.pose_align_steps):
                    pose_optimizer.zero_grad()
                    dx, drot = cam_trans_delta, cam_rot_delta
                    rot = rotation_6d_to_matrix(
                        drot + self.identity.expand(b, v, -1)
                    )  # (..., 3, 3)

                    transform = torch.eye(4, device=extrinsics.device).repeat((b, v, 1, 1))
                    transform[..., :3, :3] = rot
                    transform[..., :3, 3] = dx

                    new_extrinsics = torch.matmul(extrinsics, transform)
                    output = self.model.decoder.forward(
                        gaussians,
                        new_extrinsics,
                        batch["target"]["intrinsics"],
                        batch["target"]["near"],
                        batch["target"]["far"],
                        (h, w),
                        # cam_rot_delta=cam_rot_delta,
                        # cam_trans_delta=cam_trans_delta,
                    )

                    # Compute and log loss.
                    total_loss = 0
                    for loss_fn in self.losses:
                        loss = loss_fn.forward(output, batch, gaussians, self.global_step)
                        total_loss = total_loss + loss

                    total_loss.backward()
                    pose_optimizer.step()
                    
        # Render Gaussians.
        output = self.model.decoder.forward(
            gaussians,
            new_extrinsics,
            batch["target"]["intrinsics"],
            batch["target"]["near"],
            batch["target"]["far"],
            (h, w),
        )

        return output

    def on_test_end(self) -> None:
        name = get_cfg()["swanlab"]["name"]
        self.benchmarker.dump(self.test_cfg.output_path / name / "benchmark.json")
        self.benchmarker.dump_memory(
            self.test_cfg.output_path / name / "peak_memory.json"
        )
        self.benchmarker.summarize()

    @rank_zero_only
    def validation_step(self, batch, batch_idx, dataloader_idx=0):        
        # Render Gaussians.
        b, v, _, h, w = batch["context"]["image"].shape
        assert b == 1
        visualization_dump = {}

        encoder_output, output = self.model(batch["context"]["image"], self.global_step, visualization_dump=visualization_dump)
        depth_dict, distill_infos = encoder_output.depth_dict, encoder_output.distill_infos

        infos = encoder_output.infos

        GS_num = infos['voxelize_ratio'] * (h*w*v)
        self.log("val/GS_num", GS_num)
        
        rgb_pred = output.color[0].float()
        depth_pred = vis_depth_map(output.depth[0])

        # direct depth from gaussian means (used for visualization only)
        gaussian_means = visualization_dump["depth"][0].squeeze()
        if gaussian_means.shape[-1] == 3:
            gaussian_means = gaussian_means.mean(dim=-1)

        # Compute validation metrics.
        rgb_gt = batch["context"]["image"][0].float()
        psnr = compute_psnr(rgb_gt, rgb_pred).mean()
        self.log(f"val/psnr", psnr)
        lpips = compute_lpips(rgb_gt, rgb_pred).mean()
        self.log(f"val/lpips", lpips)
        ssim = compute_ssim(rgb_gt, rgb_pred).mean()
        self.log(f"val/ssim", ssim)

        # depth metrics
        consis_absrel = abs_relative_difference(
            rearrange(output.depth, "b v h w -> (b v) h w"),
            rearrange(depth_dict['depth'].squeeze(-1), "b v h w -> (b v) h w"),
        )
        self.log("val/consis_absrel", consis_absrel.mean())
        
        consis_delta1 = delta1_acc(
            rearrange(output.depth, "b v h w -> (b v) h w"),
            rearrange(depth_dict['depth'].squeeze(-1), "b v h w -> (b v) h w"),
            valid_mask=rearrange(torch.ones_like(output.depth, device=output.depth.device, dtype=torch.bool), "b v h w -> (b v) h w"),
        )
        self.log("val/consis_delta1", consis_delta1.mean())

        diff_map = torch.abs(output.depth - depth_dict['depth'].squeeze(-1))
        self.log("val/consis_mse", diff_map[distill_infos['conf_mask']].mean())

        # Construct comparison image.
        context_img = batch["context"]["image"][0]
        # context_img_depth = vis_depth_map(gaussian_means)
        context = []
        for i in range(context_img.shape[0]):
            context.append(context_img[i])
            # context.append(context_img_depth[i])
        
        colored_diff_map = vis_depth_map(diff_map[0], near=torch.tensor(1e-4, device=diff_map.device), far=torch.tensor(1.0, device=diff_map.device))
        model_depth_pred = depth_dict["depth"].squeeze(-1)[0]
        model_depth_pred = vis_depth_map(model_depth_pred)
        
        render_normal = (get_normal_map(output.depth.flatten(0, 1), batch["context"]["intrinsics"].flatten(0, 1)).permute(0, 3, 1, 2) + 1) / 2.
        pred_normal = (get_normal_map(depth_dict['depth'].flatten(0, 1).squeeze(-1), batch["context"]["intrinsics"].flatten(0, 1)).permute(0, 3, 1, 2) + 1) / 2.

        comparison = hcat(
            add_label(vcat(*context), "Context"),
            add_label(vcat(*rgb_pred), "Target (Prediction)"),
            add_label(vcat(*model_depth_pred), "Depth (VGGT Prediction)"),
            add_label(vcat(*depth_pred), "Depth (Prediction)"),
            add_label(vcat(*pred_normal), "Normal (VGGT Prediction)"),
            add_label(vcat(*render_normal), "Normal (Prediction)"),
            add_label(vcat(*colored_diff_map), "Diff Map"),
        )

        comparison = torch.nn.functional.interpolate(
            comparison.unsqueeze(0), 
            scale_factor=0.5, 
            mode='bicubic', 
            align_corners=False
        ).squeeze(0)
        
        self.logger.log_image(
            "comparison",
            [prep_image(comparison)],
            step=self.global_step,
            caption=batch["scene"],
        )

        # Render projections and construct projection image.
        # These are disabled for now, since RE10k scenes are effectively unbounded.

        # if isinstance(gaussians, Gaussians):
        #     projections = hcat(
        #             *render_projections(
        #                 gaussians,
        #                 256,
        #                 extra_label="",
        #             )[0]
        #         )
        #     self.logger.log_image(
        #         "projection",
        #         [prep_image(projections)],
        #         step=self.global_step,
        #     )

        # Draw cameras.
        # cameras = hcat(*render_cameras(batch, 256))
        # self.logger.log_image(
        #     "cameras", [prep_image(cameras)], step=self.global_step
        # )

        if self.encoder_visualizer is not None:
            for k, image in self.encoder_visualizer.visualize(
                batch["context"], self.global_step
            ).items():
                self.logger.log_image(k, [prep_image(image)], step=self.global_step)
        
        self.render_novel_views(batch)
        # Run video validation step. (Disabled for acceleated training)
        # self.render_video_interpolation(batch)
        # self.render_video_wobble(batch)

    @rank_zero_only
    def render_novel_views(self, batch: BatchedExample) -> None:
        _, v, _, h, w = batch["context"]["image"].shape
        enc = self.model.encoder(batch["context"]["image"], self.global_step)
        gaussians, pred = enc.gaussians, enc.pred_context_pose

        num_frames = 60
        t = torch.linspace(0, 1, num_frames, dtype=torch.float32, device=self.device)
        near = repeat(batch["context"]["near"][:, 0], "b -> b v", v=num_frames)
        far = repeat(batch["context"]["far"][:, 0], "b -> b v", v=num_frames)

        extrinsics = generate_pivot_orbit_face_initial(
            pred["extrinsic"][:, 0],
            pred["extrinsic"][:, 1],
            t,
            theta_max_deg=20,
            axis="X",                 # or "Y"
            forward_push_mean=0.02,   # 2 cm
            forward_push_std=0.02,    # 2 cm
            unit_scale=1.0,
            per_frame_noise=False     # slight breathing jitter
        )
            
        intrinsics = repeat(
            pred["intrinsic"][:, 0],
            "b i j -> b v i j", v=t.shape[0],
        )       
        output = self.model.decoder.forward(
            gaussians, extrinsics, intrinsics, near, far, (h, w), "depth"
        )
        
        images = [
            vcat(rgb, depth)
            for rgb, depth in zip(output.color[0], vis_depth_map(output.depth[0]))
        ]

        video = torch.stack(images)
        video = (video.clip(min=0, max=1) * 255).type(torch.uint8).cpu().numpy()
        video = pack([video, video[::-1][1:-1]], "* c h w")[0]
        
        video_reshaped = video.transpose(0, 2, 3, 1)
        video_reshaped = video_reshaped.astype(np.uint8)
        save_path = "novel_view_output.mp4"
        imageio.mimwrite(save_path, video_reshaped, fps=30, quality=8) 
        print(f"Video saved to: {save_path}")

    @rank_zero_only
    def render_video_wobble(self, batch: BatchedExample) -> None:
        # Two views are needed to get the wobble radius.
        _, v, _, _ = batch["context"]["extrinsics"].shape
        if v != 2:
            return

        def trajectory_fn(t):
            origin_a = batch["context"]["extrinsics"][:, 0, :3, 3]
            origin_b = batch["context"]["extrinsics"][:, 1, :3, 3]
            delta = (origin_a - origin_b).norm(dim=-1)
            extrinsics = generate_wobble(
                batch["context"]["extrinsics"][:, 0],
                delta * 0.25,
                t,
            )
            intrinsics = repeat(
                batch["context"]["intrinsics"][:, 0],
                "b i j -> b v i j",
                v=t.shape[0],
            )
            return extrinsics, intrinsics

        return self.render_video_generic(batch, trajectory_fn, "wobble", num_frames=60)

    @rank_zero_only
    def render_video_interpolation(self, batch: BatchedExample) -> None:
        _, v, _, _ = batch["context"]["extrinsics"].shape

        def trajectory_fn(t):
            extrinsics = interpolate_extrinsics(
                batch["context"]["extrinsics"][0, 0],
                batch["context"]["extrinsics"][0, 1],
                t,
            )
            intrinsics = interpolate_intrinsics(
                batch["context"]["intrinsics"][0, 0],
                batch["context"]["intrinsics"][0, 1],
                t,
            )
            return extrinsics[None], intrinsics[None]

        return self.render_video_generic(batch, trajectory_fn, "rgb")

    @rank_zero_only
    def render_video_generic(
        self,
        batch: BatchedExample,
        trajectory_fn: TrajectoryFn,
        name: str,
        num_frames: int = 30,
        smooth: bool = True,
        loop_reverse: bool = True,
    ) -> None:
        # Render probabilistic estimate of scene.
        encoder_output = self.model.encoder(batch["context"]["image"], self.global_step)
        gaussians, pred_pose = encoder_output.gaussians, encoder_output.pred_pose

        t = torch.linspace(0, 1, num_frames, dtype=torch.float32, device=self.device)
        if smooth:
            t = (torch.cos(torch.pi * (t + 1)) + 1) / 2

        extrinsics, intrinsics = trajectory_fn(t)

        _, _, _, h, w = batch["context"]["image"].shape

        # TODO: Interpolate near and far planes?
        near = repeat(batch["context"]["near"][:, 0], "b -> b v", v=num_frames)
        far = repeat(batch["context"]["far"][:, 0], "b -> b v", v=num_frames)
        output = self.model.decoder.forward(
            gaussians, extrinsics, intrinsics, near, far, (h, w), "depth"
        )
        images = [
            vcat(rgb, depth)
            for rgb, depth in zip(output.color[0], vis_depth_map(output.depth[0]))
        ]

        video = torch.stack(images)
        video = (video.clip(min=0, max=1) * 255).type(torch.uint8).cpu().numpy()
        if loop_reverse:
            video = pack([video, video[::-1][1:-1]], "* c h w")[0]
        visualizations = {f"video/{name}": ("__SWAN_VIDEO_PLACEHOLDER__")}
            
        # Since the PyTorch Lightning doesn't support video logging, log to SwanLab dashboard.
        try:
            swanlab.log  # (migrated)(visualizations)
        except Exception:
            assert isinstance(self.logger, LocalLogger)
            for key, value in visualizations.items():
                tensor = value._prepare_video(value.data)
                clip = mpy.ImageSequenceClip(list(tensor), fps=30)
                dir = LOG_PATH / key
                dir.mkdir(exist_ok=True, parents=True)
                clip.write_videofile(
                    str(dir / f"{self.global_step:0>6}.mp4"), logger=None
                )

    def print_preview_metrics(self, metrics: dict[str, float | Tensor], methods: list[str] | None = None, overlap_tag: str | None = None) -> None:
        if getattr(self, "running_metrics", None) is None:
            self.running_metrics = metrics
            self.running_metric_steps = 1
        else:
            s = self.running_metric_steps
            self.running_metrics = {
                k: ((s * v) + metrics[k]) / (s + 1)
                for k, v in self.running_metrics.items()
            }
            self.running_metric_steps += 1

        if overlap_tag is not None:
            if getattr(self, "running_metrics_sub", None) is None:
                self.running_metrics_sub = {overlap_tag: metrics}
                self.running_metric_steps_sub = {overlap_tag: 1}
            elif overlap_tag not in self.running_metrics_sub:
                self.running_metrics_sub[overlap_tag] = metrics
                self.running_metric_steps_sub[overlap_tag] = 1
            else:
                s = self.running_metric_steps_sub[overlap_tag]
                self.running_metrics_sub[overlap_tag] = {k: ((s * v) + metrics[k]) / (s + 1)
                                                         for k, v in self.running_metrics_sub[overlap_tag].items()}
                self.running_metric_steps_sub[overlap_tag] += 1

        metric_list = ["psnr", "lpips", "ssim"]

        def print_metrics(runing_metric, methods=None):
            table = []
            if methods is None:
                methods = ['ours']

            for method in methods:
                row = [
                    f"{runing_metric[f'{metric}_{method}']:.3f}"
                    for metric in metric_list
                ]
                table.append((method, *row))

            headers = ["Method"] + metric_list
            table = tabulate(table, headers)
            print(table)

        print("All Pairs:")
        print_metrics(self.running_metrics, methods)
        if overlap_tag is not None:
            for k, v in self.running_metrics_sub.items():
                print(f"Overlap: {k}")
                print_metrics(v, methods)

    def configure_optimizers(self):
        new_params, new_param_names = [], []
        pretrained_params, pretrained_param_names = [], []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            
            if "gaussian_param_head" in name or "interm" in name:
                new_params.append(param)
                new_param_names.append(name)
            else:
                pretrained_params.append(param)
                pretrained_param_names.append(name)
        
        param_dicts = [
            {
                "params": new_params,
                "lr": self.optimizer_cfg.lr,
             },
            {
                "params": pretrained_params,
                "lr": self.optimizer_cfg.lr * self.optimizer_cfg.backbone_lr_multiplier,
            },
        ]
        optimizer = torch.optim.AdamW(param_dicts, lr=self.optimizer_cfg.lr, weight_decay=0.05, betas=(0.9, 0.95))
        warm_up_steps = self.optimizer_cfg.warm_up_steps
        warm_up = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            1 / warm_up_steps,
            1,
            total_iters=warm_up_steps,
        )
        
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=get_cfg()["trainer"]["max_steps"], eta_min=self.optimizer_cfg.lr * 0.1)
        lr_scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warm_up, lr_scheduler], milestones=[warm_up_steps])

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }
