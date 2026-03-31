import copy
import utils3d

# VGGT parts
import os
import sys
from dataclasses import dataclass
from typing import List, Literal, Optional

import torch
import torch.nn.functional as F
import torchvision
from einops import rearrange
from huggingface_hub import PyTorchModelHubMixin
from jaxtyping import Float
from src.dataset.shims.bounds_shim import apply_bounds_shim
from src.dataset.shims.normalize_shim import apply_normalize_shim
from src.dataset.shims.patch_shim import apply_patch_shim
from src.dataset.types import BatchedExample, DataShim
from src.geometry.projection import sample_image_grid

from src.model.encoder.heads.vggt_dpt_gs_head import VGGT_DPT_GS_Head
from src.model.encoder.vggt.utils.geometry import (
    batchify_unproject_depth_map_to_point_map,
    unproject_depth_map_to_point_map,
)
from src.model.encoder.vggt.utils.pose_enc import pose_encoding_to_extri_intri
from src.utils.geometry import get_rel_pos  # used for model hub
from torch import nn, Tensor

from ..types import Gaussians
from .common.gaussian_adapter import (
    GaussianAdapter,
    GaussianAdapterCfg,
    UnifiedGaussianAdapter,
)
from .encoder import Encoder, EncoderOutput
from .heads import head_factory
from .visualization.encoder_visualizer_epipolar_cfg import EncoderVisualizerEpipolarCfg

root_path = os.path.abspath(".")
sys.path.append(root_path)
from src.model.encoder.heads.head_modules import TransformerBlockSelfAttn
from src.model.encoder.vggt.heads.dpt_head import DPTHead
from src.model.encoder.vggt.layers.mlp import Mlp
from src.model.encoder.vggt.models.vggt import VGGT, VGGT_v3
from src.model.encoder.pi3.models.pi3 import Pi3
from src.model.encoder.pi3.utils.geometry import homogenize_points, recover_focal_shift
from src.model.encoder.pi3.models.layers.pos_embed import RoPE2D, PositionGetter

inf = float("inf")


@dataclass
class OpacityMappingCfg:
    initial: float
    final: float
    warm_up: int


@dataclass
class GSHeadParams:
    dec_depth: int = 23
    patch_size: tuple[int, int] = (14, 14)
    enc_embed_dim: int = 2048
    dec_embed_dim: int = 2048
    feature_dim: int = 256
    depth_mode = ("exp", -inf, inf)
    conf_mode = True


@dataclass
class EncoderGenSplatCfg:
    name: Literal["gensplatpi3"]
    anchor_feat_dim: int
    voxel_size: float
    n_offsets: int
    d_feature: int
    add_view: bool
    num_monocular_samples: int
    visualizer: EncoderVisualizerEpipolarCfg
    gaussian_adapter: GaussianAdapterCfg
    apply_bounds_shim: bool
    opacity_mapping: OpacityMappingCfg
    gaussians_per_pixel: int
    num_surfaces: int
    gs_params_head_type: str
    input_mean: tuple[float, float, float] = (0.5, 0.5, 0.5)
    input_std: tuple[float, float, float] = (0.5, 0.5, 0.5)
    pretrained_weights: str = ""
    pose_free: bool = True
    pred_pose: bool = True
    gt_pose_to_pts: bool = False
    gs_prune: bool = False
    opacity_threshold: float = 0.001
    gs_keep_ratio: float = 1.0
    pred_head_type: Literal["depth", "point"] = "point"
    freeze_backbone: bool = False
    freeze_module: Literal[
        "all",
        "global",
        "frame",
        "patch_embed",
        "patch_embed+frame",
        "patch_embed+global",
        "global+frame",
        "None",
    ] = "None"
    distill: bool = True
    render_conf: bool = False
    opacity_conf: bool = False
    conf_threshold: float = 0.1
    intermediate_layer_idx: Optional[List[int]] = None
    voxelize: bool = False


def rearrange_head(feat, patch_size, H, W):
    B = feat.shape[0]
    feat = feat.transpose(-1, -2).view(B, -1, H // patch_size, W // patch_size)
    feat = F.pixel_shuffle(feat, patch_size)  # B,D,H,W
    feat = rearrange(feat, "b d h w -> b (h w) d")
    return feat


class EncoderGenSplat(Encoder[EncoderGenSplatCfg]):
    backbone: nn.Module
    gaussian_adapter: GaussianAdapter

    def __init__(self, cfg: EncoderGenSplatCfg) -> None:
        super().__init__(cfg)
        self.freeze_backbone = cfg.freeze_backbone
        self.distill = cfg.distill
        self.pred_pose = cfg.pred_pose
        model_full = Pi3.from_pretrained("yyfz233/Pi3")
        # ----------------------
        # Encoder + Decoder ≈ aggregator
        self.encoder, self.decoder = model_full.encoder, model_full.decoder 
        # ----------------------
        # Camera Pose Decoder
        self.camera_decoder, self.camera_head = model_full.camera_decoder, model_full.camera_head
        # ----------------------
        # Local Points Decoder
        self.point_decoder, self.point_head = model_full.point_decoder, model_full.point_head
        # ----------------------
        # Conf Decoder
        self.conf_decoder, self.conf_head = model_full.conf_decoder, model_full.conf_head
        # ----------------------
        #  Positonal Encoding
        # ----------------------
        self.pos_type, self.rope = 'rope100', None
        if RoPE2D is None: raise ImportError("Cannot find cuRoPE2D, please install it following the README instructions")
        freq = float(self.pos_type[len('rope'):])
        self.rope = RoPE2D(freq=freq)
        self.position_getter = PositionGetter()
        # ----------------------
        # Register_token
        # ----------------------
        num_register_tokens = 5
        self.dec_embed_dim = 1024
        self.patch_start_idx = num_register_tokens
        self.register_token = model_full.register_token
        
        self.intermediate_layer_idx = cfg.intermediate_layer_idx
        
        image_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1)
        image_std  = torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1)
        self.register_buffer("image_mean", image_mean)
        self.register_buffer("image_std", image_std)
        
        # Knowledge distillation: The parameters of the teacher network are not trained
        if self.distill:
            self.distill_encoder, self.distill_decoder = copy.deepcopy(self.encoder), copy.deepcopy(self.decoder)
            self.distill_camera_decoder, self.distill_camera_head = copy.deepcopy(self.camera_decoder), copy.deepcopy(self.camera_head)
            self.distill_point_decoder, self.distill_point_head = copy.deepcopy(self.point_decoder), copy.deepcopy(self.point_head)
            self.distill_conf_decoder, self.distill_conf_head = copy.deepcopy(self.conf_decoder), copy.deepcopy(self.conf_head)
            self.distill_register_token = copy.deepcopy(self.register_token)
            for module in [
                self.distill_encoder, self.distill_decoder,
                self.distill_camera_decoder, self.distill_camera_head,
                self.distill_point_decoder, self.distill_point_head,
                self.distill_conf_decoder, self.distill_conf_head,
                self.distill_register_token
            ]:
                if isinstance(module, nn.Module):
                    for param in module.parameters():
                        param.requires_grad = False
                        param.data = param.data.cpu()
                else:
                    module.requires_grad = False
                    module.data = module.data.cpu()

        if self.freeze_backbone:
            # Freeze backbone components
            if self.cfg.pred_head_type == "depth":
                for module in [self.aggregator, self.camera_head, self.depth_head]:
                    for param in module.parameters():
                        param.requires_grad = False
            else:
                for module in [self.aggregator, self.camera_head, self.point_head]:
                    for param in module.parameters():
                        param.requires_grad = False
        else:
            # aggregator freeze
            freeze_module = self.cfg.freeze_module
            for name, param in self.named_parameters():
                param.requires_grad = (
                    freeze_module not in name and "distill" not in name
                )

        self.pose_free = cfg.pose_free
        if self.pose_free:
            self.gaussian_adapter = UnifiedGaussianAdapter(cfg.gaussian_adapter)
        else:
            self.gaussian_adapter = GaussianAdapter(cfg.gaussian_adapter)

        self.raw_gs_dim = 1 + self.gaussian_adapter.d_in  # 1 for opacity
        self.voxel_size = cfg.voxel_size
        self.gs_params_head_type = cfg.gs_params_head_type
        # fake backbone for head parameters
        head_params = GSHeadParams()
        self.gaussian_param_head = VGGT_DPT_GS_Head(
            dim_in=2048,
            patch_size=head_params.patch_size,
            output_dim=self.raw_gs_dim + 1,
            activation="norm_exp",
            conf_activation="expp1",
            features=head_params.feature_dim,
        )

    def map_pdf_to_opacity(
        self,
        pdf: Float[Tensor, " *batch"],
        global_step: int,
    ) -> Float[Tensor, " *batch"]:
        # https://www.desmos.com/calculator/opvwti3ba9

        # Figure out the exponent.
        cfg = self.cfg.opacity_mapping
        x = cfg.initial + min(global_step / cfg.warm_up, 1) * (cfg.final - cfg.initial)
        exponent = 2**x

        # Map the probability density to an opacity.
        return 0.5 * (1 - (1 - pdf) ** exponent + pdf ** (1 / exponent))

    def normalize_pts3d(self, pts3ds, valid_masks, original_extrinsics=None):
        # normalize pts_all
        B = pts3ds.shape[0]
        pts3d_norms = []
        scale_factors = []
        for bs in range(B):
            pts3d, valid_mask = pts3ds[bs], valid_masks[bs]
            if original_extrinsics is not None:
                camera_c2w = original_extrinsics[bs]
                first_camera_w2c = (
                    camera_c2w[0].inverse().unsqueeze(0).repeat(pts3d.shape[0], 1, 1)
                )

                pts3d_homo = torch.cat(
                    [pts3d, torch.ones_like(pts3d[:, :, :, :1])], dim=-1
                )
                transformed_pts3d = torch.bmm(
                    first_camera_w2c, pts3d_homo.flatten(1, 2).transpose(1, 2)
                ).transpose(1, 2)[..., :3]
                scene_scale = torch.norm(
                    transformed_pts3d.flatten(0, 1)[valid_mask.flatten(0, 2).bool()],
                    dim=-1,
                ).mean()
            else:
                transformed_pts3d = pts3d[valid_mask]
                dis = transformed_pts3d.norm(dim=-1)
                scene_scale = dis.mean().clip(min=1e-8)
            # pts3d_norm[bs] = pts3d[bs] / scene_scale
            pts3d_norms.append(pts3d / scene_scale)
            scale_factors.append(scene_scale)
        return torch.stack(pts3d_norms, dim=0), torch.stack(scale_factors, dim=0)

    def align_pts_all_with_pts3d(
        self, pts_all, pts3d, valid_mask, original_extrinsics=None
    ):
        # align pts_all with pts3d
        B = pts_all.shape[0]

        # follow vggt's normalization implementation
        pts3d_norm, scale_factor = self.normalize_pts3d(
            pts3d, valid_mask, original_extrinsics
        )  # check if this is correct
        pts_all = pts_all * scale_factor.view(B, 1, 1, 1, 1)

        return pts_all

    def pad_tensor_list(self, tensor_list, pad_shape, value=0.0):
        padded = []
        for t in tensor_list:
            pad_len = pad_shape[0] - t.shape[0]
            if pad_len > 0:
                padding = torch.full(
                    (pad_len, *t.shape[1:]), value, device=t.device, dtype=t.dtype
                )
                t = torch.cat([t, padding], dim=0)
            padded.append(t)
        return torch.stack(padded)

    def decode(self, hidden, N, H, W, 
               use_distill: bool = False,
               dpt_head: bool = False,):
        BN, hw, _ = hidden.shape
        B = BN // N
        final_output, aggregated_output, paired_features = [], [], []
        hidden = hidden.reshape(B*N, hw, -1)
        if use_distill:
            distill_token = self.distill_register_token.repeat(B, N, 1, 1).reshape(B*N, *self.distill_register_token.shape[-2:])
            hidden = torch.cat([distill_token, hidden], dim=1)
        else:
            register_token = self.register_token.repeat(B, N, 1, 1).reshape(B*N, *self.register_token.shape[-2:])
            # Concatenate special tokens with patch tokens
            hidden = torch.cat([register_token, hidden], dim=1)
        hw = hidden.shape[1]
        if self.pos_type.startswith('rope'):
            pos = self.position_getter(B * N, H//14, W//14, hidden.device)
        if self.patch_start_idx > 0:
            # do not use position embedding for special tokens (camera and register tokens)
            # so set pos to 0 for the special tokens
            pos = pos + 1
            pos_special = torch.zeros(B * N, self.patch_start_idx, 2).to(hidden.device).to(pos.dtype)
            pos = torch.cat([pos_special, pos], dim=1)
        decoder = self.distill_decoder if use_distill else self.decoder
        if dpt_head:
            for i in range(len(decoder)):
                blk = decoder[i]
                if i % 2 == 0:
                    pos = pos.reshape(B*N, hw, -1)
                    hidden = hidden.reshape(B*N, hw, -1)
                else:
                    pos = pos.reshape(B, N*hw, -1)
                    hidden = hidden.reshape(B, N*hw, -1)
                hidden = blk(hidden, xpos=pos)
                if i+1 in [len(decoder)-1, len(decoder)]:
                    final_output.append(hidden.reshape(B*N, hw, -1))
                    
                if dpt_head and self.intermediate_layer_idx is not None:
                    if i-1 in self.intermediate_layer_idx or i in self.intermediate_layer_idx:
                        aggregated_output.append(hidden.reshape(B*N, hw, -1))
            
            for i in range(0, len(aggregated_output), 2):
                cat_feat = torch.cat([aggregated_output[i], aggregated_output[i+1]], dim=-1).reshape(B, N, hw, -1)
                paired_features.append(cat_feat)            
                
            return torch.cat([final_output[0], final_output[1]], dim=-1), pos.reshape(B*N, hw, -1), paired_features
        else:
            for i in range(len(decoder)):
                blk = decoder[i]
                if i % 2 == 0:
                    pos = pos.reshape(B*N, hw, -1)
                    hidden = hidden.reshape(B*N, hw, -1)
                else:
                    pos = pos.reshape(B, N*hw, -1)
                    hidden = hidden.reshape(B, N*hw, -1)
                hidden = blk(hidden, xpos=pos)
                if i+1 in [len(decoder)-1, len(decoder)]:
                    final_output.append(hidden.reshape(B*N, hw, -1))   
                
            return torch.cat([final_output[0], final_output[1]], dim=-1), pos.reshape(B*N, hw, -1)

    def forward(
        self,
        image: torch.Tensor,
        global_step: int = 0,
        visualization_dump: Optional[dict] = None,
    ) -> Gaussians:
        original_image = image.clone()
        image = (image - self.image_mean) / self.image_std
        device = image.device
        b, v, _, h, w = image.shape
        distill_infos = {}
        # 1. Generation of Distilled Priors. All distill_ variables are pre-trained VGGT prediction results and can be regarded as pseudo labels
        if self.distill:
            distill_image = image.clone().detach()
            for module in [
                self.distill_encoder, self.distill_decoder,
                self.distill_camera_decoder, self.distill_camera_head,
                self.distill_point_decoder, self.distill_point_head,
                self.distill_conf_decoder, self.distill_conf_head,
                self.distill_register_token
            ]:
                if isinstance(module, nn.Module):
                    for param in module.parameters():
                        param.data = param.data.to(device, non_blocking=True)
                else:
                    module.data = module.data.to(device, non_blocking=True)

            with torch.no_grad():
                with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
                    distill_image = distill_image.reshape(b*v, _, h, w).to(torch.bfloat16)
                    distill_hidden = self.distill_encoder(distill_image, is_training=True)
                    if isinstance(distill_hidden, dict):
                        distill_hidden = distill_hidden["x_norm_patchtokens"]
                    distill_hidden, pos = self.decode(distill_hidden, v, h, w, use_distill=True)
                    
                    distill_point_hidden = self.distill_point_decoder(distill_hidden, xpos=pos)
                    distill_camera_hidden = self.distill_camera_decoder(distill_hidden, xpos=pos)
                    distill_conf_hidden = self.distill_conf_decoder(distill_hidden, xpos=pos)

                # Process with default precision
                with torch.amp.autocast("cuda", enabled=False):
                    # Get local points
                    distill_point_hidden = distill_point_hidden.to(torch.float32)
                    ret = self.distill_point_head([distill_point_hidden[:, self.patch_start_idx:]], (h, w)).reshape(b, v, h, w, -1)
                    xy, z = ret.split([2, 1], dim=-1)
                    z = torch.exp(z)
                    distill_local_points = torch.cat([xy * z, z], dim=-1)
                    
                    # Get camera pose
                    distill_camera_hidden = distill_camera_hidden.to(torch.float32)
                    distill_camera_poses = self.distill_camera_head(
                        distill_camera_hidden[:, self.patch_start_idx:], 
                        h//14, w//14).reshape(b, v, 4, 4)                 # [B, V, 4, 4])
                    # unproject local points using camera poses
                    distill_points = torch.einsum(
                        'bnij, bnhwj -> bnhwi',
                        distill_camera_poses, 
                        homogenize_points(distill_local_points))[..., :3] # [B, V, H, W, 3]
                    
                    # Get confidence
                    distill_conf_hidden = distill_conf_hidden.float()
                    distill_conf = self.distill_conf_head([distill_conf_hidden[:, self.patch_start_idx:]], (h, w)).reshape(b, v, h, w)

                # Store results
                distill_infos["camera_poses"] = distill_camera_poses
                distill_infos["point_map"] = distill_points
                conf_threshold = torch.quantile(
                    distill_conf.flatten(2, 3).float(), 0.3, dim=-1, keepdim=True
                )  # Get threshold for each view
                conf_mask = distill_conf > conf_threshold.unsqueeze(-1)
                distill_infos["conf_mask"] = conf_mask         # [B, V, H, W]

                for module in [
                    self.distill_encoder, self.distill_decoder,
                    self.distill_camera_decoder, self.distill_camera_head,
                    self.distill_point_decoder, self.distill_point_head,
                    self.distill_conf_decoder, self.distill_conf_head,
                    self.distill_register_token
                ]:
                    if isinstance(module, nn.Module):
                        for param in module.parameters():
                            param.data = param.data.cpu()
                    else:
                        module.data = module.data.cpu()
                # Clean up to save memory
                del distill_hidden, pos
                del distill_point_hidden, distill_camera_hidden, distill_conf_hidden
                del ret, xy, z, distill_local_points
                del distill_camera_poses, distill_points, distill_conf
                torch.cuda.empty_cache()

        # 2. Student backbone: Multi-view aggregation
        with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
            image = image.reshape(b*v, _, h, w).to(torch.bfloat16)
            hidden = self.encoder(image, is_training=True)
            if isinstance(hidden, dict):
                hidden = hidden["x_norm_patchtokens"]
            hidden, pos, aggregated_list = self.decode(hidden, v, h, w, dpt_head=True)

        # 3. Student heads: pose head + geometry head (point/depth)
        point_hidden = self.point_decoder(hidden, xpos=pos)
        camera_hidden = self.camera_decoder(hidden, xpos=pos)
        conf_hidden = self.conf_decoder(hidden, xpos=pos)
        
        with torch.amp.autocast("cuda", enabled=False):  
            # local points
            point_hidden = point_hidden.float()
            ret = self.point_head([point_hidden[:, self.patch_start_idx:]], (h, w)).reshape(b, v, h, w, -1)
            xy, z = ret.split([2, 1], dim=-1)
            z = torch.exp(z)
            local_points = torch.cat([xy * z, z], dim=-1)

            # confidence
            conf_hidden = conf_hidden.float()
            point_conf = self.conf_head([conf_hidden[:, self.patch_start_idx:]], (h, w)).reshape(b, v, h, w)
                        
            # camera
            camera_hidden = camera_hidden.float()
            camera_poses = self.camera_head(camera_hidden[:, self.patch_start_idx:], h//14, w//14).reshape(b, v, 4, 4)

            # unproject local points using camera poses
            points = torch.einsum('bnij, bnhwj -> bnhwi', camera_poses, homogenize_points(local_points))[..., :3]
            
            if self.cfg.render_conf:
                conf_valid = torch.quantile(
                    point_conf.flatten(0, 1), self.cfg.conf_threshold
                )
                conf_valid_mask = point_conf > conf_valid
            else:
                conf_valid_mask = torch.ones_like(point_conf, dtype=torch.bool)

        # 4. Student's Gaussian parameter head: regress token features + 3D points into "pixel-level anchor features"
        # dpt style gs_head input format out.shape = (B, V, 84, H, W)
        out = self.gaussian_param_head(
            aggregated_list,
            points.flatten(0, 1).permute(0, 3, 1, 2),
            original_image.reshape(b, v, -1, h, w),
            patch_start_idx=self.patch_start_idx,
            image_size=(h, w),
        )

        del aggregated_list
        torch.cuda.empty_cache()

        # 5. Scene scale estimation (monitor only): for printing/alignment reference
        pts_flat = points.flatten(2, 3)
        scene_scale = pts_flat.norm(dim=-1).mean().clip(min=1e-8)

        anchor_feats, gs_conf = out[:, :, : self.raw_gs_dim], out[:, :, self.raw_gs_dim]

        # 6. (Optional) Pixel →  Voxel Fusion: Redundancy Reduction & Learning-Friendly
        neural_feats_list, neural_pts_list = [], []
        for b_i in range(b):
            neural_feats_list.append(
                anchor_feats[b_i].permute(0, 2, 3, 1)[conf_valid_mask[b_i]]
            )
            neural_pts_list.append(points[b_i][conf_valid_mask[b_i]])

        max_voxels = max(f.shape[0] for f in neural_feats_list)
        neural_feats = self.pad_tensor_list(neural_feats_list, (max_voxels,), -1e10)
        neural_pts = self.pad_tensor_list(neural_pts_list, (max_voxels,), -1e4)

        depths = neural_pts[..., -1].unsqueeze(-1)
        densities = neural_feats[..., 0].sigmoid()

        assert len(densities.shape) == 2, "the shape of densities should be (B, N)"
        assert neural_pts.shape[1] > 1, "the number of voxels should be greater than 1"

        opacity = self.map_pdf_to_opacity(densities, global_step).squeeze(-1)
        if self.cfg.opacity_conf:
            shift = torch.quantile(point_conf.float(), self.cfg.conf_threshold)
            opacity = opacity * torch.sigmoid(point_conf.float() - shift)[
                conf_valid_mask
            ].unsqueeze(0)

        # GS Prune, but only works when bs = 1
        # if want to support bs > 1, need to random prune gaussians based on the rank of opacity like LongLRM
        # Note: we not prune gaussians here, but we will try it in the future
        if self.cfg.gs_prune and b == 1:
            opacity_threshold = self.cfg.opacity_threshold
            gaussian_usage = opacity > opacity_threshold  # (B, N)

            print(
                f"based on opacity threshold {opacity_threshold}, pruned {gaussian_usage.shape[1] - neural_pts.shape[1]} gaussians out of {gaussian_usage.shape[1]}"
            )

            if (gaussian_usage.sum() / gaussian_usage.numel()) > self.cfg.gs_keep_ratio:
                # rank by opacity
                num_keep = int(gaussian_usage.shape[1] * self.cfg.gs_keep_ratio)
                idx_sort = opacity.argsort(dim=1, descending=True)
                keep_idx = idx_sort[:, :num_keep]
                gaussian_usage = torch.zeros_like(gaussian_usage, dtype=torch.bool)
                gaussian_usage.scatter_(1, keep_idx, True)

            neural_pts = neural_pts[gaussian_usage].view(b, -1, 3).contiguous()
            depths = depths[gaussian_usage].view(b, -1, 1).contiguous()
            neural_feats = (
                neural_feats[gaussian_usage].view(b, -1, self.raw_gs_dim).contiguous()
            )
            opacity = opacity[gaussian_usage].view(b, -1).contiguous()

            print(
                f"finally pruned {gaussian_usage.shape[1] - neural_pts.shape[1]} gaussians out of {gaussian_usage.shape[1]}"
            )

        gaussians = self.gaussian_adapter.forward(
            neural_pts,
            depths,
            opacity,
            neural_feats[..., 1:].squeeze(2),
        )

        if visualization_dump is not None:
            visualization_dump["depth"] = rearrange(
                points[..., -1].flatten(2, 3).unsqueeze(-1).unsqueeze(-1),
                "b v (h w) srf s -> b v h w srf s",
                h=h, w=w,
            )

        infos = {}
        infos["scene_scale"] = scene_scale
        infos["voxelize_ratio"] = densities.shape[1] / (h * w * v)

        masks = torch.sigmoid(point_conf) > 0.1
        aspect_ratio = w / h
        focal, shift_val = recover_focal_shift(local_points, masks)
        fx, fy = focal / 2 * (1 + aspect_ratio ** 2) ** 0.5 / aspect_ratio, focal / 2 * (1 + aspect_ratio ** 2) ** 0.5
        intrinsic = utils3d.torch.intrinsics_from_focal_center(fx, fy, torch.tensor(0.5, device=device), torch.tensor(0.5, device=device))

        return EncoderOutput(
            gaussians=gaussians,
            pred_pose_enc_list=[camera_poses],
            pred_context_pose=dict(
                extrinsic=camera_poses,
                intrinsic=intrinsic,
            ),
            depth_dict=dict(depth=local_points[..., 2:3], conf_valid_mask=conf_valid_mask),
            infos=infos,
            distill_infos=distill_infos,
        )

    def get_data_shim(self) -> DataShim:
        def data_shim(batch: BatchedExample) -> BatchedExample:
            batch = apply_normalize_shim(
                batch,
                self.cfg.input_mean,
                self.cfg.input_std,
            )

            return batch

        return data_shim
