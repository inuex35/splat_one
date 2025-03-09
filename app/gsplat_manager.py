#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import json
import math
import time
import numpy as np
import torch
import torch.nn.functional as F
import yaml
import tqdm
import imageio
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union
from typing_extensions import Literal, assert_never

from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QTreeWidget, QTreeWidgetItem
)
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt, QThread, pyqtSignal

from scipy.spatial.transform import Rotation  # 回転変換用
from pyproj import Proj

import nerfview
from utils.logger import setup_logger
from utils.datasets.opensfm import *  # Camera, Image, angle_axis_to_quaternion, Parser, Dataset など
from utils.gsplat_utils.lib_bilagrid import (
    BilateralGrid,
    slice,
    color_correct,
    total_variation_loss,
)
from gsplat.compression import PngCompression
from gsplat.distributed import cli
from gsplat.rendering import rasterization
from gsplat.strategy import DefaultStrategy, MCMCStrategy
from gsplat.optimizers import SelectiveAdam
from utils.gsplat_utils.utils import AppearanceOptModule, CameraOptModule, knn, rgb_to_sh, set_random_seed
import viser
import tyro
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from fused_ssim import fused_ssim
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

logger = setup_logger()

# -----------------------------------------------------
# 共通のユーティリティ関数
# -----------------------------------------------------

def load_reconstruction(file_path: str):
    """JSONファイルから再構築データを読み込む関数"""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        if "reconstructions" in data:
            return data["reconstructions"]
        return data
    except FileNotFoundError:
        logger.error(f"File {file_path} not found.")
        return None
    except json.JSONDecodeError:
        logger.error(f"File {file_path} is not valid JSON.")
        return None

def qvec2rotmat(qvec):
    return np.array([
        [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]])

# -----------------------------------------------------
# トレーニング関連のコード（gsplat_manager_training.py の内容）
# -----------------------------------------------------

@dataclass
class Config:
    disable_viewer: bool = True
    ckpt: Optional[List[str]] = None
    compression: Optional[Literal["png"]] = None
    render_traj_path: str = "interp"
    data_dir: str = "data_dir"
    data_factor: int = 4
    result_dir: str = "results/"
    test_every: int = 8
    patch_size: Optional[int] = None
    global_scale: float = 1.0
    normalize_world_space: bool = True
    camera_model: Literal["pinhole", "ortho", "fisheye", "spherical"] = "spherical"
    port: int = 8080
    batch_size: int = 1
    steps_scaler: float = 1.0
    max_steps: int = 30000
    eval_steps: List[int] = field(default_factory=lambda: [7000, 30000])
    save_steps: List[int] = field(default_factory=lambda: [7000, 30000])
    init_type: str = "sfm"
    init_num_pts: int = 100000
    init_extent: float = 3.0
    sh_degree: int = 3
    sh_degree_interval: int = 1000
    init_opa: float = 0.1
    init_scale: float = 1.0
    ssim_lambda: float = 0.2
    near_plane: float = 0.01
    far_plane: float = 1e8
    strategy: Union[DefaultStrategy, MCMCStrategy] = field(default_factory=DefaultStrategy)
    packed: bool = False
    sparse_grad: bool = False
    visible_adam: bool = False
    antialiased: bool = False
    random_bkgd: bool = False
    opacity_reg: float = 0.0
    scale_reg: float = 0.0
    pose_opt: bool = False
    pose_opt_lr: float = 1e-5
    pose_opt_reg: float = 1e-6
    pose_noise: float = 0.0
    app_opt: bool = False
    app_embed_dim: int = 16
    app_opt_lr: float = 1e-3
    app_opt_reg: float = 1e-6
    use_bilateral_grid: bool = False
    bilateral_grid_shape: Tuple[int, int, int] = (16, 16, 8)
    depth_loss: bool = False
    depth_lambda: float = 1e-2
    tb_every: int = 100
    tb_save_image: bool = False
    lpips_net: Literal["vgg", "alex"] = "alex"

    def adjust_steps(self, factor: float):
        self.eval_steps = [int(i * factor) for i in self.eval_steps]
        self.save_steps = [int(i * factor) for i in self.save_steps]
        self.max_steps = int(self.max_steps * factor)
        self.sh_degree_interval = int(self.sh_degree_interval * factor)
        strategy = self.strategy
        if isinstance(strategy, DefaultStrategy):
            strategy.refine_start_iter = int(strategy.refine_start_iter * factor)
            strategy.refine_stop_iter = int(strategy.refine_stop_iter * factor)
            strategy.reset_every = int(strategy.reset_every * factor)
            strategy.refine_every = int(strategy.refine_every * factor)
        elif isinstance(strategy, MCMCStrategy):
            strategy.refine_start_iter = int(strategy.refine_start_iter * factor)
            strategy.refine_stop_iter = int(strategy.refine_stop_iter * factor)
            strategy.refine_every = int(strategy.refine_every * factor)
        else:
            assert_never(strategy)

def create_splats_with_optimizers(
    parser: Parser,
    init_type: str = "sfm",
    init_num_pts: int = 100000,
    init_extent: float = 3.0,
    init_opacity: float = 1e-1,
    init_scale: float = 1.0,
    scene_scale: float = 1.0,
    sh_degree: int = 3,
    sparse_grad: bool = False,
    visible_adam: bool = False,
    batch_size: int = 1,
    feature_dim: Optional[int] = None,
    device: str = "cuda",
    world_rank: int = 0,
    world_size: int = 1,
) -> Tuple[torch.nn.ParameterDict, Dict[str, torch.optim.Optimizer]]:
    if init_type == "sfm":
        points = torch.from_numpy(parser.points).float()
        rgbs = torch.from_numpy(parser.points_rgb / 255.0).float()
    elif init_type == "random":
        points = init_extent * scene_scale * (torch.rand((init_num_pts, 3)) * 2 - 1)
        rgbs = torch.rand((init_num_pts, 3))
    else:
        raise ValueError("Please specify a correct init_type: sfm or random")

    dist2_avg = (knn(points, 4)[:, 1:] ** 2).mean(dim=-1)
    dist_avg = torch.sqrt(dist2_avg)
    scales = torch.log(dist_avg * init_scale).unsqueeze(-1).repeat(1, 3)
    points = points[world_rank::world_size]
    rgbs = rgbs[world_rank::world_size]
    scales = scales[world_rank::world_size]
    N = points.shape[0]
    quats = torch.rand((N, 4))
    opacities = torch.logit(torch.full((N,), init_opacity))
    params = [
        ("means", torch.nn.Parameter(points), 1.6e-4 * scene_scale),
        ("scales", torch.nn.Parameter(scales), 5e-3),
        ("quats", torch.nn.Parameter(quats), 1e-3),
        ("opacities", torch.nn.Parameter(opacities), 5e-2),
    ]
    if feature_dim is None:
        colors = torch.zeros((N, (sh_degree + 1) ** 2, 3))
        colors[:, 0, :] = rgb_to_sh(rgbs)
        params.append(("sh0", torch.nn.Parameter(colors[:, :1, :]), 2.5e-3))
        params.append(("shN", torch.nn.Parameter(colors[:, 1:, :]), 2.5e-3 / 20))
    else:
        features = torch.rand(N, feature_dim)
        params.append(("features", torch.nn.Parameter(features), 2.5e-3))
        colors = torch.logit(rgbs)
        params.append(("colors", torch.nn.Parameter(colors), 2.5e-3))
    splats = torch.nn.ParameterDict({n: v for n, v, _ in params}).to(device)
    BS = batch_size * world_size
    if sparse_grad:
        optimizer_class = torch.optim.SparseAdam
    elif visible_adam:
        optimizer_class = SelectiveAdam
    else:
        optimizer_class = torch.optim.Adam
    optimizers = {
        name: optimizer_class(
            [{"params": splats[name], "lr": lr * math.sqrt(BS), "name": name}],
            eps=1e-15 / math.sqrt(BS),
            betas=(1 - BS * (1 - 0.9), 1 - BS * (1 - 0.999)),
        )
        for name, _, lr in params
    }
    return splats, optimizers

class Runner:
    """Engine for training and testing."""
    def __init__(self, local_rank: int, world_rank, world_size: int, cfg: Config) -> None:
        set_random_seed(42 + local_rank)
        self.cfg = cfg
        self.world_rank = world_rank
        self.local_rank = local_rank
        self.world_size = world_size
        self.device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
        os.makedirs(cfg.result_dir, exist_ok=True)
        self.ckpt_dir = os.path.join(cfg.result_dir, "ckpts")
        os.makedirs(self.ckpt_dir, exist_ok=True)
        self.stats_dir = os.path.join(cfg.result_dir, "stats")
        os.makedirs(self.stats_dir, exist_ok=True)
        self.render_dir = os.path.join(cfg.result_dir, "renders")
        os.makedirs(self.render_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=os.path.join(cfg.result_dir, "tb"))
        self.parser = Parser(
            data_dir=cfg.data_dir,
            factor=cfg.data_factor,
            normalize=cfg.normalize_world_space
        )
        self.trainset = Dataset(
            self.parser,
            split="train",
            patch_size=cfg.patch_size,
            load_depths=cfg.depth_loss,
        )
        self.allset = Dataset(
            self.parser,
            split="all",
            patch_size=cfg.patch_size,
            load_depths=cfg.depth_loss,
        )
        self.valset = Dataset(self.parser, split="val")
        self.scene_scale = self.parser.scene_scale * 1.1 * cfg.global_scale
        print("Scene scale:", self.scene_scale)
        feature_dim = 32 if cfg.app_opt else None
        self.splats, self.optimizers = create_splats_with_optimizers(
            self.parser,
            init_type=cfg.init_type,
            init_num_pts=cfg.init_num_pts,
            init_extent=cfg.init_extent,
            init_opacity=cfg.init_opa,
            init_scale=cfg.init_scale,
            scene_scale=self.scene_scale,
            sh_degree=cfg.sh_degree,
            sparse_grad=cfg.sparse_grad,
            visible_adam=cfg.visible_adam,
            batch_size=cfg.batch_size,
            feature_dim=feature_dim,
            device=self.device,
            world_rank=world_rank,
            world_size=world_size,
        )
        print("Model initialized. Number of GS:", len(self.splats["means"]))
        self.cfg.strategy.check_sanity(self.splats, self.optimizers)
        if isinstance(self.cfg.strategy, DefaultStrategy):
            self.strategy_state = self.cfg.strategy.initialize_state(
                scene_scale=self.scene_scale
            )
        elif isinstance(self.cfg.strategy, MCMCStrategy):
            self.strategy_state = self.cfg.strategy.initialize_state()
        else:
            assert_never(self.cfg.strategy)
        self.compression_method = None
        if cfg.compression is not None:
            if cfg.compression == "png":
                self.compression_method = PngCompression()
            else:
                raise ValueError(f"Unknown compression strategy: {cfg.compression}")
        self.pose_optimizers = []
        if cfg.pose_opt:
            self.pose_adjust = CameraOptModule(len(self.trainset)).to(self.device)
            self.pose_adjust.zero_init()
            self.pose_optimizers = [
                torch.optim.Adam(
                    self.pose_adjust.parameters(),
                    lr=cfg.pose_opt_lr * math.sqrt(cfg.batch_size),
                    weight_decay=cfg.pose_opt_reg,
                )
            ]
            if world_size > 1:
                self.pose_adjust = DDP(self.pose_adjust)
        if cfg.pose_noise > 0.0:
            self.pose_perturb = CameraOptModule(len(self.trainset)).to(self.device)
            self.pose_perturb.random_init(cfg.pose_noise)
            if world_size > 1:
                self.pose_perturb = DDP(self.pose_perturb)
        self.app_optimizers = []
        if cfg.app_opt:
            assert feature_dim is not None
            self.app_module = AppearanceOptModule(
                len(self.trainset), feature_dim, cfg.app_embed_dim, cfg.sh_degree
            ).to(self.device)
            torch.nn.init.zeros_(self.app_module.color_head[-1].weight)
            torch.nn.init.zeros_(self.app_module.color_head[-1].bias)
            self.app_optimizers = [
                torch.optim.Adam(
                    self.app_module.embeds.parameters(),
                    lr=cfg.app_opt_lr * math.sqrt(cfg.batch_size) * 10.0,
                    weight_decay=cfg.app_opt_reg,
                ),
                torch.optim.Adam(
                    self.app_module.color_head.parameters(),
                    lr=cfg.app_opt_lr * math.sqrt(cfg.batch_size),
                ),
            ]
            if world_size > 1:
                self.app_module = DDP(self.app_module)
        self.bil_grid_optimizers = []
        if cfg.use_bilateral_grid:
            self.bil_grids = BilateralGrid(
                len(self.trainset),
                grid_X=cfg.bilateral_grid_shape[0],
                grid_Y=cfg.bilateral_grid_shape[1],
                grid_W=cfg.bilateral_grid_shape[2],
            ).to(self.device)
            self.bil_grid_optimizers = [
                torch.optim.Adam(
                    self.bil_grids.parameters(),
                    lr=2e-3 * math.sqrt(cfg.batch_size),
                    eps=1e-15,
                ),
            ]
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(self.device)
        self.psnr = PeakSignalNoiseRatio(data_range=1.0).to(self.device)
        if cfg.lpips_net == "alex":
            self.lpips = LearnedPerceptualImagePatchSimilarity(
                net_type="alex", normalize=True
            ).to(self.device)
        elif cfg.lpips_net == "vgg":
            self.lpips = LearnedPerceptualImagePatchSimilarity(
                net_type="vgg", normalize=False
            ).to(self.device)
        else:
            raise ValueError(f"Unknown LPIPS network: {cfg.lpips_net}")

    def rasterize_splats(
        self,
        camtoworlds: Tensor,
        Ks: Tensor,
        width: int,
        height: int,
        masks: Optional[Tensor] = None,
        **kwargs,
    ) -> Tuple[Tensor, Tensor, Dict]:
        means = self.splats["means"]
        quats = self.splats["quats"]
        scales = torch.exp(self.splats["scales"])
        opacities = torch.sigmoid(self.splats["opacities"])
        image_ids = kwargs.pop("image_ids", None)
        if self.cfg.app_opt:
            colors = self.app_module(
                features=self.splats["features"],
                embed_ids=image_ids,
                dirs=means[None, :, :] - camtoworlds[:, None, :3, 3],
                sh_degree=kwargs.pop("sh_degree", self.cfg.sh_degree),
            )
            colors = colors + self.splats["colors"]
            colors = torch.sigmoid(colors)
        else:
            colors = torch.cat([self.splats["sh0"], self.splats["shN"]], 1)
        rasterize_mode = "antialiased" if self.cfg.antialiased else "classic"
        render_colors, render_alphas, info = rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=colors,
            viewmats=torch.linalg.inv(camtoworlds),
            Ks=Ks,
            width=width,
            height=height,
            packed=self.cfg.packed,
            absgrad=(self.cfg.strategy.absgrad if isinstance(self.cfg.strategy, DefaultStrategy) else False),
            sparse_grad=self.cfg.sparse_grad,
            rasterize_mode=rasterize_mode,
            distributed=self.world_size > 1,
            camera_model=self.cfg.camera_model,
            **kwargs,
        )
        if masks is not None:
            render_colors[~masks] = 0
        return render_colors, render_alphas, info

    def train(self):
        cfg = self.cfg
        device = self.device
        world_rank = self.world_rank
        world_size = self.world_size
        if world_rank == 0:
            with open(os.path.join(cfg.result_dir, "cfg.yml"), "w") as f:
                yaml.dump(vars(cfg), f)
        max_steps = cfg.max_steps
        init_step = 0
        schedulers = [
            torch.optim.lr_scheduler.ExponentialLR(
                self.optimizers["means"], gamma=0.01 ** (1.0 / max_steps)
            ),
        ]
        if cfg.pose_opt:
            schedulers.append(
                torch.optim.lr_scheduler.ExponentialLR(
                    self.pose_optimizers[0], gamma=0.01 ** (1.0 / max_steps)
                )
            )
        if cfg.use_bilateral_grid:
            schedulers.append(
                torch.optim.lr_scheduler.ChainedScheduler(
                    [
                        torch.optim.lr_scheduler.LinearLR(
                            self.bil_grid_optimizers[0],
                            start_factor=0.01,
                            total_iters=1000,
                        ),
                        torch.optim.lr_scheduler.ExponentialLR(
                            self.bil_grid_optimizers[0], gamma=0.01 ** (1.0 / max_steps)
                        ),
                    ]
                )
            )
        trainloader = torch.utils.data.DataLoader(
            self.trainset,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=4,
            persistent_workers=True,
            pin_memory=True,
        )
        trainloader_iter = iter(trainloader)
        global_tic = time.time()
        pbar = tqdm.tqdm(range(init_step, max_steps))
        for step in pbar:
            try:
                data = next(trainloader_iter)
            except StopIteration:
                trainloader_iter = iter(trainloader)
                data = next(trainloader_iter)
            camtoworlds = data["camtoworld"].to(device)
            Ks = data["K"].to(device)
            pixels = data["image"].to(device) / 255.0
            num_train_rays_per_step = pixels.shape[0] * pixels.shape[1] * pixels.shape[2]
            image_ids = data["image_id"].to(device)
            masks = data["mask"].to(device) if "mask" in data else None
            if cfg.depth_loss:
                points = data["points"].to(device)
                depths_gt = data["depths"].to(device)
            height, width = pixels.shape[1:3]
            if cfg.pose_noise:
                camtoworlds = self.pose_perturb(camtoworlds, image_ids)
            if cfg.pose_opt:
                camtoworlds = self.pose_adjust(camtoworlds, image_ids)
            sh_degree_to_use = min(step // cfg.sh_degree_interval, cfg.sh_degree)
            renders, alphas, info = self.rasterize_splats(
                camtoworlds=camtoworlds,
                Ks=Ks,
                width=width,
                height=height,
                sh_degree=sh_degree_to_use,
                near_plane=cfg.near_plane,
                far_plane=cfg.far_plane,
                image_ids=image_ids,
                render_mode="RGB+ED" if cfg.depth_loss else "RGB",
                masks=masks,
            )
            if renders.shape[-1] == 4:
                colors, depths = renders[..., 0:3], renders[..., 3:4]
            else:
                colors, depths = renders, None
            if cfg.use_bilateral_grid:
                grid_y, grid_x = torch.meshgrid(
                    (torch.arange(height, device=device) + 0.5) / height,
                    (torch.arange(width, device=device) + 0.5) / width,
                    indexing="ij",
                )
                grid_xy = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)
                colors = slice(self.bil_grids, grid_xy, colors, image_ids)["rgb"]
            if cfg.random_bkgd:
                bkgd = torch.rand(1, 3, device=device)
                colors = colors + bkgd * (1.0 - alphas)
            self.cfg.strategy.step_pre_backward(
                params=self.splats,
                optimizers=self.optimizers,
                state=self.strategy_state,
                step=step,
                info=info,
            )
            l1loss = F.l1_loss(colors, pixels)
            ssimloss = 1.0 - fused_ssim(
                colors.permute(0, 3, 1, 2), pixels.permute(0, 3, 1, 2), padding="valid"
            )
            loss = l1loss * (1.0 - cfg.ssim_lambda) + ssimloss * cfg.ssim_lambda
            if cfg.depth_loss:
                points = torch.stack(
                    [
                        points[:, :, 0] / (width - 1) * 2 - 1,
                        points[:, :, 1] / (height - 1) * 2 - 1,
                    ],
                    dim=-1,
                )
                grid = points.unsqueeze(2)
                depths = F.grid_sample(
                    depths.permute(0, 3, 1, 2), grid, align_corners=True
                )
                depths = depths.squeeze(3).squeeze(1)
                disp = torch.where(depths > 0.0, 1.0 / depths, torch.zeros_like(depths))
                disp_gt = 1.0 / depths_gt
                depthloss = F.l1_loss(disp, disp_gt) * self.scene_scale
                loss += depthloss * cfg.depth_lambda
            if cfg.use_bilateral_grid:
                tvloss = 10 * total_variation_loss(self.bil_grids.grids)
                loss += tvloss
            if cfg.opacity_reg > 0.0:
                loss = loss + cfg.opacity_reg * torch.abs(torch.sigmoid(self.splats["opacities"])).mean()
            if cfg.scale_reg > 0.0:
                loss = loss + cfg.scale_reg * torch.abs(torch.exp(self.splats["scales"])).mean()
            loss.backward()
            desc = f"loss={loss.item():.3f}| sh degree={sh_degree_to_use}| "
            if cfg.depth_loss:
                desc += f"depth loss={depthloss.item():.6f}| "
            if cfg.pose_opt and cfg.pose_noise:
                pose_err = F.l1_loss(camtoworlds, camtoworlds)
                desc += f"pose err={pose_err.item():.6f}| "
            pbar.set_description(desc)
            if world_rank == 0 and cfg.tb_every > 0 and step % cfg.tb_every == 0:
                mem = torch.cuda.max_memory_allocated() / 1024**3
                self.writer.add_scalar("train/loss", loss.item(), step)
                self.writer.add_scalar("train/l1loss", l1loss.item(), step)
                self.writer.add_scalar("train/ssimloss", ssimloss.item(), step)
                self.writer.add_scalar("train/num_GS", len(self.splats["means"]), step)
                self.writer.add_scalar("train/mem", mem, step)
                if cfg.depth_loss:
                    self.writer.add_scalar("train/depthloss", depthloss.item(), step)
                if cfg.use_bilateral_grid:
                    self.writer.add_scalar("train/tvloss", tvloss.item(), step)
                if cfg.tb_save_image:
                    canvas = torch.cat([pixels, colors], dim=2).detach().cpu().numpy()
                    canvas = canvas.reshape(-1, *canvas.shape[2:])
                    self.writer.add_image("train/render", canvas, step)
                self.writer.flush()
            if step in [i - 1 for i in cfg.save_steps] or step == max_steps - 1:
                mem = torch.cuda.max_memory_allocated() / 1024**3
                stats = {
                    "mem": mem,
                    "ellipse_time": time.time() - global_tic,
                    "num_GS": len(self.splats["means"]),
                }
                print("Step: ", step, stats)
                with open(os.path.join(self.stats_dir, f"train_step{step:04d}_rank{world_rank}.json"), "w") as f:
                    json.dump(stats, f)
                data = {"step": step, "splats": self.splats.state_dict()}
                if cfg.pose_opt:
                    if world_size > 1:
                        data["pose_adjust"] = self.pose_adjust.module.state_dict()
                    else:
                        data["pose_adjust"] = self.pose_adjust.state_dict()
                if cfg.app_opt:
                    if world_size > 1:
                        data["app_module"] = self.app_module.module.state_dict()
                    else:
                        data["app_module"] = self.app_module.state_dict()
                torch.save(data, os.path.join(self.ckpt_dir, f"ckpt_{step}_rank{world_rank}.pt"))
            if cfg.sparse_grad:
                assert cfg.packed, "Sparse gradients only work with packed mode."
                gaussian_ids = info["gaussian_ids"]
                for k in self.splats.keys():
                    grad = self.splats[k].grad
                    if grad is None or grad.is_sparse:
                        continue
                    self.splats[k].grad = torch.sparse_coo_tensor(
                        indices=gaussian_ids[None],
                        values=grad[gaussian_ids],
                        size=self.splats[k].size(),
                        is_coalesced=len(Ks) == 1,
                    )
            if cfg.visible_adam:
                if cfg.packed:
                    visibility_mask = torch.zeros_like(self.splats["opacities"], dtype=bool)
                    visibility_mask.scatter_(0, info["gaussian_ids"], 1)
                else:
                    visibility_mask = (info["radii"] > 0).any(0)
            for optimizer in self.optimizers.values():
                if cfg.visible_adam:
                    optimizer.step(visibility_mask)
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            for optimizer in self.pose_optimizers:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            for optimizer in self.app_optimizers:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            for optimizer in self.bil_grid_optimizers:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            for scheduler in schedulers:
                scheduler.step()
            if isinstance(self.cfg.strategy, DefaultStrategy):
                self.cfg.strategy.step_post_backward(
                    params=self.splats,
                    optimizers=self.optimizers,
                    state=self.strategy_state,
                    step=step,
                    info=info,
                    packed=cfg.packed,
                )
            elif isinstance(self.cfg.strategy, MCMCStrategy):
                self.cfg.strategy.step_post_backward(
                    params=self.splats,
                    optimizers=self.optimizers,
                    state=self.strategy_state,
                    step=step,
                    info=info,
                    lr=schedulers[0].get_last_lr()[0],
                )
            else:
                assert_never(self.cfg.strategy)
            if step in [i - 1 for i in cfg.eval_steps]:
                self.eval(step)
                self.render_traj(step)
            if cfg.compression is not None and step in [i - 1 for i in cfg.eval_steps]:
                self.run_compression(step=step)
        if not cfg.disable_viewer:
            print("Viewer running... Ctrl+C to exit.")
            time.sleep(1000000)

    @torch.no_grad()
    def eval(self, step: int, stage: str = "val"):
        print("Running evaluation...")
        cfg = self.cfg
        device = self.device
        world_rank = self.world_rank
        world_size = self.world_size
        valloader = torch.utils.data.DataLoader(
            self.valset, batch_size=1, shuffle=False, num_workers=1
        )
        ellipse_time = 0
        metrics = defaultdict(list)
        for i, data in enumerate(valloader):
            camtoworlds = data["camtoworld"].to(device)
            Ks = data["K"].to(device)
            pixels = data["image"].to(device) / 255.0
            masks = data["mask"].to(device) if "mask" in data else None
            height, width = pixels.shape[1:3]
            torch.cuda.synchronize()
            tic = time.time()
            colors, _, _ = self.rasterize_splats(
                camtoworlds=camtoworlds,
                Ks=Ks,
                width=width,
                height=height,
                sh_degree=cfg.sh_degree,
                near_plane=cfg.near_plane,
                far_plane=cfg.far_plane,
                masks=masks,
            )
            torch.cuda.synchronize()
            ellipse_time += time.time() - tic
            colors = torch.clamp(colors, 0.0, 1.0)
            canvas_list = [pixels, colors]
            if world_rank == 0:
                canvas = torch.cat(canvas_list, dim=2).squeeze(0).cpu().numpy()
                canvas = (canvas * 255).astype(np.uint8)
                imageio.imwrite(os.path.join(self.render_dir, f"{stage}_step{step}_{i:04d}.png"), canvas)
                pixels_p = pixels.permute(0, 3, 1, 2)
                colors_p = colors.permute(0, 3, 1, 2)
                metrics["psnr"].append(self.psnr(colors_p, pixels_p))
                metrics["ssim"].append(self.ssim(colors_p, pixels_p))
                metrics["lpips"].append(self.lpips(colors_p, pixels_p))
                if cfg.use_bilateral_grid:
                    cc_colors = color_correct(colors, pixels)
                    cc_colors_p = cc_colors.permute(0, 3, 1, 2)
                    metrics["cc_psnr"].append(self.psnr(cc_colors_p, pixels_p))
        if world_rank == 0:
            ellipse_time /= len(valloader)
            stats = {k: torch.stack(v).mean().item() for k, v in metrics.items()}
            stats.update({"ellipse_time": ellipse_time, "num_GS": len(self.splats["means"])})
            print(f"PSNR: {stats['psnr']:.3f}, SSIM: {stats['ssim']:.4f}, LPIPS: {stats['lpips']:.3f} Time: {stats['ellipse_time']:.3f}s/image Number of GS: {stats['num_GS']}")
            with open(os.path.join(self.stats_dir, f"{stage}_step{step:04d}.json"), "w") as f:
                json.dump(stats, f)
            for k, v in stats.items():
                self.writer.add_scalar(f"{stage}/{k}", v, step)
            self.writer.flush()

    @torch.no_grad()
    def render_traj(self, step: int):
        print("Running trajectory rendering...")
        cfg = self.cfg
        device = self.device
        camtoworlds_all = self.parser.camtoworlds[5:-5]
        if cfg.render_traj_path == "interp":
            camtoworlds_all = generate_interpolated_path(camtoworlds_all, 1)
        elif cfg.render_traj_path == "ellipse":
            height = camtoworlds_all[:, 2, 3].mean()
            camtoworlds_all = generate_ellipse_path_z(camtoworlds_all, height=height)
        elif cfg.render_traj_path == "spiral":
            camtoworlds_all = generate_spiral_path(
                camtoworlds_all,
                bounds=self.parser.bounds * self.scene_scale,
                spiral_scale_r=self.parser.extconf["spiral_radius_scale"],
            )
        else:
            raise ValueError(f"Render trajectory type not supported: {cfg.render_traj_path}")
        camtoworlds_all = np.concatenate(
            [
                camtoworlds_all,
                np.repeat(np.array([[[0.0, 0.0, 0.0, 1.0]]]), len(camtoworlds_all), axis=0),
            ],
            axis=1,
        )
        camtoworlds_all = torch.from_numpy(camtoworlds_all).float().to(device)
        K = torch.from_numpy(list(self.parser.Ks_dict.values())[0]).float().to(device)
        width, height = list(self.parser.imsize_dict.values())[0]
        video_dir = os.path.join(cfg.result_dir, "videos")
        os.makedirs(video_dir, exist_ok=True)
        writer = imageio.get_writer(os.path.join(video_dir, f"traj_{step}.mp4"), fps=30)
        for i in tqdm.tqdm(range(len(camtoworlds_all)), desc="Rendering trajectory"):
            camtoworlds = camtoworlds_all[i : i + 1]
            Ks = K[None]
            renders, _, _ = self.rasterize_splats(
                camtoworlds=camtoworlds,
                Ks=Ks,
                width=width,
                height=height,
                sh_degree=cfg.sh_degree,
                near_plane=cfg.near_plane,
                far_plane=cfg.far_plane,
                render_mode="RGB+ED",
            )
            colors = torch.clamp(renders[..., 0:3], 0.0, 1.0)
            depths = renders[..., 3:4]
            depths = (depths - depths.min()) / (depths.max() - depths.min())
            canvas_list = [colors, depths.repeat(1, 1, 1, 3)]
            canvas = torch.cat(canvas_list, dim=2).squeeze(0).cpu().numpy()
            canvas = (canvas * 255).astype(np.uint8)
            writer.append_data(canvas)
        writer.close()
        print(f"Video saved to {video_dir}/traj_{step}.mp4")

    @torch.no_grad()
    def run_compression(self, step: int):
        print("Running compression...")
        world_rank = self.world_rank
        compress_dir = os.path.join(self.cfg.result_dir, "compression", f"rank{world_rank}")
        os.makedirs(compress_dir, exist_ok=True)
        self.compression_method.compress(compress_dir, self.splats)
        splats_c = self.compression_method.decompress(compress_dir)
        for k in splats_c.keys():
            self.splats[k].data = splats_c[k].to(self.device)
        self.eval(step=step, stage="compress")

    @torch.no_grad()
    def _viewer_render_fn(self, camera_state: nerfview.CameraState, img_wh: Tuple[int, int]):
        W, H = img_wh
        c2w = camera_state.c2w
        K = camera_state.get_K(img_wh)
        c2w = torch.from_numpy(c2w).float().to(self.device)
        K = torch.from_numpy(K).float().to(self.device)
        render_colors, _, _ = self.rasterize_splats(
            camtoworlds=c2w[None],
            Ks=K[None],
            width=W,
            height=H,
            sh_degree=self.cfg.sh_degree,
            radius_clip=3.0,
        )
        return render_colors[0].cpu().numpy()

# -----------------------------------------------------
# GUI統合部：GsplatManager（レンダリングおよびトレーニング処理を含む）
# -----------------------------------------------------

class TrainingThread(QThread):
    training_finished = pyqtSignal()

    def __init__(self, runner: Runner, parent=None):
        super().__init__(parent)
        self.runner = runner

    def run(self):
        self.runner.train()
        self.training_finished.emit()

class GsplatManager(QWidget):
    def __init__(self, work_dir, parent=None):
        super().__init__(parent)
        self.work_dir = work_dir
        # JSONファイルから再構築データを読み込むパス
        self.reconstruction_json_path = os.path.join(self.work_dir, "reconstruction.json")
        # Config と Runner の初期化
        cfg = Config(
            data_dir=self.work_dir,
            result_dir=os.path.join(self.work_dir, "results"),
            disable_viewer=True,
            max_steps=30000  # GUI上では簡易レンダリングのために短いステップ数を設定
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.runner = Runner(local_rank=0, world_rank=0, world_size=1, cfg=cfg)
        # OpenSfM の再構築データからカメラ・画像情報を取得
        self.read_opensfm()
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("Gsplat")
        main_layout = QVBoxLayout()
        # レンダリング結果表示用ラベル
        self.image_label = QLabel("Rendering result")
        self.image_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(self.image_label)
        # ボタン配置レイアウト
        button_layout = QHBoxLayout()
        self.render_button = QPushButton("Render")
        self.render_button.clicked.connect(self.update_render)
        button_layout.addWidget(self.render_button)
        self.train_button = QPushButton("Training start")
        self.train_button.clicked.connect(self.start_training)
        button_layout.addWidget(self.train_button)
        main_layout.addLayout(button_layout)
        self.setLayout(main_layout)

    def read_opensfm(self):
        reconstructions = load_reconstruction(self.reconstruction_json_path)
        if reconstructions is None:
            logger.error("再構築データが読み込めませんでした。")
            return
        # ここでは最初の再構築データのみを利用する例
        self.images = {}
        i = 0
        reference_lat_0 = reconstructions[0]["reference_lla"]["latitude"]
        reference_lon_0 = reconstructions[0]["reference_lla"]["longitude"]
        reference_alt_0 = reconstructions[0]["reference_lla"]["altitude"]
        e2u_zone = int(divmod(reference_lon_0, 6)[0]) + 31
        e2u_conv = Proj(proj='utm', zone=e2u_zone, ellps='WGS84')
        reference_x_0, reference_y_0 = e2u_conv(reference_lon_0, reference_lat_0)
        if reference_lat_0 < 0:
            reference_y_0 += 10000000
        self.cameras = {}
        self.camera_names = {}
        cam_id = 1
        for reconstruction in reconstructions:
            for i, camera in enumerate(reconstruction["cameras"]):
                camera_name = camera
                camera_info = reconstruction["cameras"][camera]
                if camera_info['projection_type'] in ['spherical', 'equirectangular']:
                    camera_id = 0
                    model = "SPHERICAL"
                    width = reconstruction["cameras"][camera]["width"]
                    height = reconstruction["cameras"][camera]["height"]
                    params = np.array([0])
                    self.cameras[camera_id] = Camera(id=camera_id, model=model, width=width, height=height, params=params, panorama=True)
                    self.camera_names[camera_name] = camera_id
                elif reconstruction["cameras"][camera]['projection_type'] == "perspective":
                    model = "SIMPLE_PINHOLE"
                    width = reconstruction["cameras"][camera]["width"]
                    height = reconstruction["cameras"][camera]["height"]
                    f = reconstruction["cameras"][camera]["focal"] * width
                    k1 = reconstruction["cameras"][camera]["k1"]
                    k2 = reconstruction["cameras"][camera]["k2"]
                    params = np.array([f, width / 2, height / 2, k1, k2])
                    camera_id = cam_id
                    self.cameras[camera_id] = Camera(id=camera_id, model=model, width=width, height=height, params=params, panorama=False)
                    self.camera_names[camera_name] = camera_id
                    cam_id += 1
            reference_lat = reconstruction["reference_lla"]["latitude"]
            reference_lon = reconstruction["reference_lla"]["longitude"]
            reference_alt = reconstruction["reference_lla"]["altitude"]
            reference_x, reference_y = e2u_conv(reference_lon, reference_lat)
            if reference_lat < 0:
                reference_y += 10000000
            for j, shot in enumerate(reconstruction["shots"]):
                translation = reconstruction["shots"][shot]["translation"]
                rotation = reconstruction["shots"][shot]["rotation"]
                qvec = angle_axis_to_quaternion(rotation)
                diff_ref_x = reference_x - reference_x_0
                diff_ref_y = reference_y - reference_y_0
                diff_ref_alt = reference_alt - reference_alt_0
                tvec = np.array([translation[0], translation[1], translation[2]])
                diff_ref = np.array([diff_ref_x, diff_ref_y, diff_ref_alt])
                camera_name = reconstruction["shots"][shot]["camera"]
                camera_id = self.camera_names.get(camera_name, 0)
                image_id = j
                image_name = shot
                xys = np.array([0, 0])
                point3D_ids = np.array([0, 0])
                self.images[image_id] = Image(id=image_id, qvec=qvec, tvec=tvec, camera_id=camera_id, name=image_name, xys=xys, point3D_ids=point3D_ids, diff_ref=diff_ref)

    def move_to_camera(self, image_name):
        # Runner の allset から全インデックスを取得
        for idx in self.runner.allset.indices:
            # Runner の Parser に格納されている画像情報を取得
            camera_image = self.runner.parser.images[idx]
            if camera_image.name == image_name:
                logger.info(f"Moving to camera position for image '{image_name}'")
                # 画像ごとに保持している tvec, qvec を用いて c2w 行列を構築
                position = camera_image.tvec
                rotation_matrix = qvec2rotmat(camera_image.qvec)
                c2w = np.eye(4)
                c2w[:3, :3] = rotation_matrix.T
                c2w[:3, 3] = position / 100  # 例：スケール調整

                # Scipy を用いて回転行列からクォータニオンを取得
                r = Rotation.from_matrix(rotation_matrix)
                qvec = r.as_quat()  # [x, y, z, w] の順（ライブラリによって順序は異なる場合あり）
                logger.info(f"Computed quaternion: {qvec}")

                # nerfview の CameraState を生成
                camera_state = nerfview.CameraState(
                    fov=90,
                    aspect=1.0,
                    c2w=c2w,
                )
                img_wh = (640, 480)
                # GPU でのレンダリング用にテンソル変換
                c2w_tensor = torch.from_numpy(camera_state.c2w).float().to(self.device)
                K = camera_state.get_K(img_wh)
                K_tensor = torch.from_numpy(K).float().to(self.device)

                # eval と同様に、near_plane, far_plane などのパラメータを渡してレンダリング
                render, _, _ = self.runner.rasterize_splats(
                    camtoworlds=c2w_tensor[None],
                    Ks=K_tensor[None],
                    width=img_wh[0],
                    height=img_wh[1],
                    sh_degree=self.runner.cfg.sh_degree,
                    near_plane=self.runner.cfg.near_plane,
                    far_plane=self.runner.cfg.far_plane,
                    radius_clip=3.0,
                )
                # 0〜1 にクランプして uint8 へ変換
                render = torch.clamp(render, 0.0, 1.0)
                render_uint8 = (render.cpu().detach().numpy() * 255).astype(np.uint8)
                render_uint8 = np.squeeze(render_uint8, axis=0)
                h, w, channels = render_uint8.shape
                bytes_per_line = channels * w
                qimage = QImage(render_uint8.data, w, h, bytes_per_line, QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(qimage)
                self.image_label.setPixmap(pixmap)
                break


    def update_render(self):
        # ここでは、最初の画像のカメラ位置でレンダリングを更新する例です
        if self.images:
            first_image = list(self.images.values())[0]
            self.move_to_camera(first_image.name)
        else:
            logger.error("画像情報が存在しません。")

    def start_training(self):
        self.train_button.setEnabled(False)
        self.training_thread = TrainingThread(self.runner)
        self.training_thread.training_finished.connect(self.on_training_finished)
        self.training_thread.start()
        logger.info("トレーニングを開始しました。")

    def on_training_finished(self):
        self.train_button.setEnabled(True)
        logger.info("トレーニングが完了しました。")

    def on_camera_image_tree_double_click(self, image_name):
        self.move_to_camera(image_name)

# -----------------------------------------------------
# メイン処理
# -----------------------------------------------------

if __name__ == "__main__":
    app = QApplication(sys.argv)
    work_dir = "./workdir"  # 作業ディレクトリ（適宜変更してください）
    manager = GsplatManager(work_dir)
    manager.resize(800, 600)
    manager.show()
    sys.exit(app.exec_())
