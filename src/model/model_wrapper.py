from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Protocol, runtime_checkable
import time
import yaml
import sys
sys.path.append('../../')

import moviepy.editor as mpy
import torch
import wandb
from einops import pack, rearrange, repeat
from jaxtyping import Float
from pytorch_lightning import LightningModule
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.utilities import rank_zero_only
from torch import Tensor, nn, optim
from torch.nn import functional as F

from ..dataset.data_module import get_data_shim
from ..dataset.types import BatchedExample
from ..evaluation.metrics import compute_lpips, compute_psnr, compute_ssim
from ..global_cfg import get_cfg
from ..loss import Loss
from ..misc.benchmarker import Benchmarker
from ..misc.image_io import prep_image, save_image
from ..misc.LocalLogger import LOG_PATH, LocalLogger
from ..misc.step_tracker import StepTracker
from ..visualization.annotation import add_label
from ..visualization.camera_trajectory.interpolation import (
    interpolate_extrinsics,
    interpolate_intrinsics,
)
from ..visualization.camera_trajectory.wobble import (
    generate_wobble,
    generate_wobble_transformation,
)
from ..visualization.color_map import apply_color_map_to_image
from ..visualization.layout import add_border, hcat, vcat
from ..visualization.validation_in_3d import render_cameras, render_projections
from .decoder.decoder import Decoder, DepthRenderingMode
from .encoder import Encoder
from .encoder.visualization.encoder_visualizer import EncoderVisualizer
from .decoder.decoder_latent import DecoderLatent, DecoderLatentTiny, DecoderLatentTinyWithNorm


@dataclass
class OptimizerCfg:
    lr: float
    warm_up_steps: int


@dataclass
class TestCfg:
    output_path: Path


@dataclass
class TrainCfg:
    depth_mode: DepthRenderingMode | None
    extended_visualization: bool
    depth_sample_deterministic: bool


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
    logger: Optional[WandbLogger]
    encoder: nn.Module
    encoder_visualizer: Optional[EncoderVisualizer]
    decoder: Decoder
    decoder_latent: nn.Module
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
        encoder: Encoder,
        encoder_visualizer: Optional[EncoderVisualizer],
        decoder: Decoder,
        losses: list[Loss],
        step_tracker: StepTracker | None, 
        decoder_latent: nn.Module | None,
    ) -> None:
        super().__init__()
        self.optimizer_cfg = optimizer_cfg
        self.test_cfg = test_cfg
        self.train_cfg = train_cfg
        self.step_tracker = step_tracker
        
        # Set up the model.
        self.encoder = encoder
        self.encoder_visualizer = encoder_visualizer
        self.decoder = decoder
        self.decoder_latent = decoder_latent
        self.data_shim = get_data_shim(self.encoder)
        self.losses = nn.ModuleList(losses)

        self.downsample = self.decoder_latent.upsample
        self.d_latent = self.decoder.d_latent

        # This is used for testing.
        self.benchmarker = Benchmarker()

        # # Calculate average inference times
        # self.inference_times = []

    def training_step(self, batch, batch_idx):
        batch: BatchedExample = self.data_shim(batch)
        _, _, _, h, w = batch["target"]["image"].shape

        # Run the model.
        if self.d_latent == 3:
            gaussians = self.encoder(batch["context"], self.global_step, deterministic=self.train_cfg.depth_sample_deterministic)
        elif self.d_latent in [4, 5, 6]:
            gaussians, gaussians_2 = self.encoder(batch["context"], self.global_step, deterministic=self.train_cfg.depth_sample_deterministic)
        else:
            raise ValueError(f"Invalid d_latent: {self.d_latent}")

        if self.decoder_latent is not None:
            h_new, w_new = h // self.downsample, w // self.downsample    # downsample
        else:
            h_new, w_new = h, w

        output = self.decoder.forward(
            gaussians,
            batch["target"]["extrinsics"],
            batch["target"]["intrinsics"],
            batch["target"]["near"],
            batch["target"]["far"],
            (h_new, w_new),
            depth_mode=self.train_cfg.depth_mode,
        )
        
        if self.d_latent in [4, 5, 6]:
            output_2 = self.decoder.forward(
                gaussians_2,
                batch["target"]["extrinsics"],
                batch["target"]["intrinsics"],
                batch["target"]["near"],
                batch["target"]["far"],
                (h_new, w_new),
                depth_mode=self.train_cfg.depth_mode,
            )
            output.color = torch.cat((output.color, output_2.color), dim=2)[:, :, :self.d_latent]

        # Latent decoder
        b, v, _, _, _ = output.color.shape
        output.color = rearrange(output.color, "b v c h w -> (b v) c h w")

        if isinstance(self.decoder_latent, DecoderLatent):    # Input channels: 3, output channels: 3
            output.color = self.decoder_latent.forward(output.color)
            output.color = (output.color - output.color.min()) / (output.color.max() - output.color.min())
            output.color = rearrange(output.color, "(b v) c h w -> b v c h w", b=b, v=v)

        elif isinstance(self.decoder_latent, DecoderLatentTiny):    # Input channels: 4, output channels: 3
            output.color = self.decoder_latent.forward(output.color)
            output.color = rearrange(output.color, "(b v) c h w -> b v c h w", b=b, v=v)
        
        elif isinstance(self.decoder_latent, DecoderLatentTinyWithNorm):    # Input channels: 4, output channels: 3
            output.color = self.decoder_latent.forward(output.color)
            output.color = rearrange(output.color, "(b v) c h w -> b v c h w", b=b, v=v)
        
        else: # No latent decoder
            output.color = rearrange(output.color, "(b v) c h w -> b v c h w", b=b, v=v)


        target_gt = batch["target"]["image"]

        # Compute metrics.
        psnr_probabilistic = compute_psnr(
            rearrange(target_gt, "b v c h w -> (b v) c h w"),
            rearrange(output.color, "b v c h w -> (b v) c h w"),
        )
        self.log("train/psnr_probabilistic", psnr_probabilistic.mean())

        # Compute and log loss.
        total_loss = 0
        for loss_fn in self.losses:
            loss = loss_fn.forward(output, batch, gaussians, self.global_step)
            self.log(f"loss/{loss_fn.name}", loss)
            total_loss = total_loss + loss
        self.log("loss/total", total_loss)

        if self.global_rank == 0:
            print(
                f"train step {self.global_step}; "
                f"scene = {batch['scene']}; "
                f"context = {batch['context']['index'].tolist()}; "
                f"loss = {total_loss:.6f}"
            )

        # Tell the data loader processes about the current step.
        if self.step_tracker is not None:
            self.step_tracker.set_step(self.global_step)

        return total_loss

    def test_step(self, batch, batch_idx):
        batch: BatchedExample = self.data_shim(batch)

        # torch.cuda.synchronize()
        # t0 = time.time()

        b, v, _, h, w = batch["target"]["image"].shape
        assert b == 1
        if batch_idx % 100 == 0:
            print(f"Test step {batch_idx:0>6}.")

        # Render Gaussians.

        if self.d_latent == 3:
            with self.benchmarker.time("encoder"):
                gaussians = self.encoder(
                    batch["context"],
                    self.global_step,
                    deterministic=self.train_cfg.depth_sample_deterministic,
                )
        elif self.d_latent in [4, 5, 6]:
            with self.benchmarker.time("encoder"):
                gaussians, gaussians_2 = self.encoder(
                    batch["context"],
                    self.global_step,
                    deterministic=self.train_cfg.depth_sample_deterministic,
                )
        else:
            raise ValueError(f"Invalid d_latent: {self.d_latent}")
        
        # torch.cuda.synchronize()
        # t_encoder = time.time() - t0
        # torch.cuda.synchronize()
        # t0 = time.time()

        if self.decoder_latent is not None:
            h_new, w_new = h // self.downsample, w // self.downsample    # downsample
        else:
            h_new, w_new = h, w

        with self.benchmarker.time("decoder", num_calls=v):
            output = self.decoder.forward(
                gaussians,
                batch["target"]["extrinsics"],
                batch["target"]["intrinsics"],
                batch["target"]["near"],
                batch["target"]["far"],
                (h_new, w_new),
            )
        
        if self.d_latent in [4, 5, 6]:
            with self.benchmarker.time("decoder", num_calls=v):
                output_2 = self.decoder.forward(
                    gaussians_2,
                    batch["target"]["extrinsics"],
                    batch["target"]["intrinsics"],
                    batch["target"]["near"],
                    batch["target"]["far"],
                    (h_new, w_new),
                )
                output.color = torch.cat((output.color, output_2.color), dim=2)[:, :, :self.d_latent]

        # torch.cuda.synchronize()
        # t_splatting = time.time() - t0
        # torch.cuda.synchronize()
        # t0 = time.time() 

        # Latent decoder
        b, v, _, _, _ = output.color.shape
        output.color = rearrange(output.color, "b v c h w -> (b v) c h w")

        if isinstance(self.decoder_latent, DecoderLatent):    # Input channels: 3, output channels: 3
            output.color = self.decoder_latent.forward(output.color)
            # output.color = F.interpolate(output.color, size=(h, w), mode="bilinear", align_corners=False)
            output.color = (output.color - output.color.min()) / (output.color.max() - output.color.min())
            output.color = rearrange(output.color, "(b v) c h w -> b v c h w", b=b, v=v)

        elif isinstance(self.decoder_latent, DecoderLatentTiny):    # Input channels: 4, output channels: 3
            output.color = self.decoder_latent.forward(output.color)
            output.color = rearrange(output.color, "(b v) c h w -> b v c h w", b=b, v=v)
        
        elif isinstance(self.decoder_latent, DecoderLatentTinyWithNorm):    # Input channels: 4, output channels: 3
            output.color = self.decoder_latent.forward(output.color)
            output.color = rearrange(output.color, "(b v) c h w -> b v c h w", b=b, v=v)
        
        else: # No latent decoder
            output.color = rearrange(output.color, "(b v) c h w -> b v c h w", b=b, v=v)
        
        # torch.cuda.synchronize()
        # t_decoder_latent = time.time() - t0
        
        # torch.cuda.synchronize()
        # t_inf = time.time() - t0

        # if batch_idx > 0:
        #     self.inference_times.append(t_inf)
        #     avg_t_inf = torch.tensor(self.inference_times).mean()
        #     print(f"Average inference time: {avg_t_inf:.6f} s")

        # Save images.
        (scene,) = batch["scene"]
        name = get_cfg()["wandb"]["name"]
        path = self.test_cfg.output_path / name
        for index, color in zip(batch["target"]["index"][0], output.color[0]):
            save_image(color, path / scene / f"color/{index:0>6}.png")
        for index, color in zip(
            batch["context"]["index"][0], batch["context"]["image"][0]
        ):
            save_image(color, path / scene / f"context/{index:0>6}.png")


    def on_test_end(self) -> None:
        name = get_cfg()["wandb"]["name"]
        self.benchmarker.dump(self.test_cfg.output_path / name / "benchmark.json")
        self.benchmarker.dump_memory(
            self.test_cfg.output_path / name / "peak_memory.json"
        )

    @rank_zero_only
    def validation_step(self, batch, batch_idx):
        batch: BatchedExample = self.data_shim(batch)
        
        if self.global_rank == 0:
            print(
                f"validation step {self.global_step}; "
                f"scene = {batch['scene']}; "
                f"context = {batch['context']['index'].tolist()}"
            )

        # Render Gaussians.
        b, _, _, h, w = batch["target"]["image"].shape
        assert b == 1

        if self.d_latent == 3:
            gaussians_probabilistic = self.encoder(
                batch["context"],
                self.global_step,
                deterministic=False,
            )
        elif self.d_latent in [4, 5, 6]:
            gaussians_probabilistic, gaussians_probabilistic_2 = self.encoder(
                batch["context"],
                self.global_step,
                deterministic=False,
            )
        else:
            raise ValueError(f"Invalid d_latent: {self.d_latent}")

        if self.decoder_latent is not None:
            h_new, w_new = h // self.downsample, w // self.downsample    # downsample
        else:
            h_new, w_new = h, w

        output_probabilistic = self.decoder.forward(
            gaussians_probabilistic,
            batch["target"]["extrinsics"],
            batch["target"]["intrinsics"],
            batch["target"]["near"],
            batch["target"]["far"],
            (h_new, w_new),
        )

        if self.d_latent in [4, 5, 6]:
            output_probabilistic_2 = self.decoder.forward(
                gaussians_probabilistic_2,
                batch["target"]["extrinsics"],
                batch["target"]["intrinsics"],
                batch["target"]["near"],
                batch["target"]["far"],
                (h_new, w_new),
            )
            output_probabilistic.color = torch.cat((output_probabilistic.color, output_probabilistic_2.color), dim=2)[:, :, :self.d_latent]

        # Latent decoder
        b, v, _, _, _ = output_probabilistic.color.shape
        output_probabilistic.color = rearrange(output_probabilistic.color, "b v c h w -> (b v) c h w")

        if isinstance(self.decoder_latent, DecoderLatent):    # Input channels: 3, output channels: 3
            output_probabilistic.color = self.decoder_latent.forward(output_probabilistic.color)
            output_probabilistic.color = (output_probabilistic.color - output_probabilistic.color.min()) / (output_probabilistic.color.max() - output_probabilistic.color.min())
            output_probabilistic.color = rearrange(output_probabilistic.color, "(b v) c h w -> b v c h w", b=b, v=v)

        elif isinstance(self.decoder_latent, DecoderLatentTiny):    # Input channels: 4, output channels: 3
            output_probabilistic.color = self.decoder_latent.forward(output_probabilistic.color)
            output_probabilistic.color = rearrange(output_probabilistic.color, "(b v) c h w -> b v c h w", b=b, v=v)
        
        elif isinstance(self.decoder_latent, DecoderLatentTinyWithNorm):    # Input channels: 4, output channels: 3
            output_probabilistic.color = self.decoder_latent.forward(output_probabilistic.color)
            output_probabilistic.color = rearrange(output_probabilistic.color, "(b v) c h w -> b v c h w", b=b, v=v)
        
        else: # No latent decoder
            output_probabilistic.color = rearrange(output_probabilistic.color, "(b v) c h w -> b v c h w", b=b, v=v)


        rgb_probabilistic = output_probabilistic.color[0]

        if self.d_latent == 3:
            gaussians_deterministic = self.encoder(
                batch["context"],
                self.global_step,
                deterministic=True,
            )
        elif self.d_latent in [4, 5, 6]:
            gaussians_deterministic, gaussians_deterministic_2 = self.encoder(
                batch["context"],
                self.global_step,
                deterministic=True,
            )
        else:
            raise ValueError(f"Invalid d_latent: {self.d_latent}")

        output_deterministic = self.decoder.forward(
            gaussians_deterministic,
            batch["target"]["extrinsics"],
            batch["target"]["intrinsics"],
            batch["target"]["near"],
            batch["target"]["far"],
            (h_new, w_new),
        )

        if self.d_latent in [4, 5, 6]:
            output_deterministic_2 = self.decoder.forward(
                gaussians_deterministic_2,
                batch["target"]["extrinsics"],
                batch["target"]["intrinsics"],
                batch["target"]["near"],
                batch["target"]["far"],
                (h_new, w_new),
            )
            output_deterministic.color = torch.cat((output_deterministic.color, output_deterministic_2.color), dim=2)[:, :, :self.d_latent]

        # Latent decoder
        b, v, _, _, _ = output_deterministic.color.shape
        output_deterministic.color = rearrange(output_deterministic.color, "b v c h w -> (b v) c h w")

        if isinstance(self.decoder_latent, DecoderLatent):    # Input channels: 3, output channels: 3
            output_deterministic.color = self.decoder_latent.forward(output_deterministic.color)
            output_deterministic.color = (output_deterministic.color - output_deterministic.color.min()) / (output_deterministic.color.max() - output_deterministic.color.min())
            output_deterministic.color = rearrange(output_deterministic.color, "(b v) c h w -> b v c h w", b=b, v=v)

        elif isinstance(self.decoder_latent, DecoderLatentTiny):    # Input channels: 4, output channels: 3
            output_deterministic.color = self.decoder_latent.forward(output_deterministic.color)
            output_deterministic.color = rearrange(output_deterministic.color, "(b v) c h w -> b v c h w", b=b, v=v)
        
        elif isinstance(self.decoder_latent, DecoderLatentTinyWithNorm):    # Input channels: 4, output channels: 3
            output_deterministic.color = self.decoder_latent.forward(output_deterministic.color)
            output_deterministic.color = rearrange(output_deterministic.color, "(b v) c h w -> b v c h w", b=b, v=v)
        
        else: # No latent decoder
            output_deterministic.color = rearrange(output_deterministic.color, "(b v) c h w -> b v c h w", b=b, v=v)


        rgb_deterministic = output_deterministic.color[0]

        # Compute validation metrics.
        rgb_gt = batch["target"]["image"][0]
        for tag, rgb in zip(
            ("deterministic", "probabilistic"), (rgb_deterministic, rgb_probabilistic)
        ):
            psnr = compute_psnr(rgb_gt, rgb).mean()
            self.log(f"val/psnr_{tag}", psnr)
            lpips = compute_lpips(rgb_gt, rgb).mean()
            self.log(f"val/lpips_{tag}", lpips)
            ssim = compute_ssim(rgb_gt, rgb).mean()
            self.log(f"val/ssim_{tag}", ssim)

        # Construct comparison image.
        comparison = hcat(
            add_label(vcat(*batch["context"]["image"][0]), "Context"),
            add_label(vcat(*rgb_gt), "Target (Ground Truth)"),
            add_label(vcat(*rgb_probabilistic), "Target (Probabilistic)"),
            add_label(vcat(*rgb_deterministic), "Target (Deterministic)"),
        )
        self.logger.log_image(
            "comparison",
            [prep_image(add_border(comparison))],
            step=self.global_step,
            caption=batch["scene"],
        )

        # Render projections and construct projection image.
        # These are disabled for now, since RE10k scenes are effectively unbounded.
        projections = vcat(
            hcat(
                *render_projections(
                    gaussians_probabilistic,
                    256,
                    extra_label="(Probabilistic)",
                )[0]
            ),
            hcat(
                *render_projections(
                    gaussians_deterministic, 
                    256, 
                    extra_label="(Deterministic)",
                )[0]
            ),
            align="left",
        )
        self.logger.log_image(
            "projection",
            [prep_image(add_border(projections))],
            step=self.global_step,
        )

        # Draw cameras.
        cameras = hcat(*render_cameras(batch, 256))
        self.logger.log_image(
            "cameras", [prep_image(add_border(cameras))], step=self.global_step
        )
        
        # if self.encoder_visualizer is not None:
        #     for k, image in self.encoder_visualizer.visualize(
        #         batch["context"], self.global_step
        #     ).items():
        #         self.logger.log_image(k, [prep_image(image)], step=self.global_step)

        # # Run video validation step.
        # self.render_video_interpolation(batch)
        # self.render_video_wobble(batch)
        # if self.train_cfg.extended_visualization:
        #     self.render_video_interpolation_exaggerated(batch)

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
                (
                    batch["context"]["extrinsics"][0, 1]
                    if v == 2
                    else batch["target"]["extrinsics"][0, 0]
                ),
                t,
            )
            intrinsics = interpolate_intrinsics(
                batch["context"]["intrinsics"][0, 0],
                (
                    batch["context"]["intrinsics"][0, 1]
                    if v == 2
                    else batch["target"]["intrinsics"][0, 0]
                ),
                t,
            )
            return extrinsics[None], intrinsics[None]

        return self.render_video_generic(batch, trajectory_fn, "rgb")

    @rank_zero_only
    def render_video_interpolation_exaggerated(self, batch: BatchedExample) -> None:
        # Two views are needed to get the wobble radius.
        _, v, _, _ = batch["context"]["extrinsics"].shape
        if v != 2:
            return

        def trajectory_fn(t):
            origin_a = batch["context"]["extrinsics"][:, 0, :3, 3]
            origin_b = batch["context"]["extrinsics"][:, 1, :3, 3]
            delta = (origin_a - origin_b).norm(dim=-1)
            tf = generate_wobble_transformation(
                delta * 0.5,
                t,
                5,
                scale_radius_with_t=False,
            )
            extrinsics = interpolate_extrinsics(
                batch["context"]["extrinsics"][0, 0],
                (
                    batch["context"]["extrinsics"][0, 1]
                    if v == 2
                    else batch["target"]["extrinsics"][0, 0]
                ),
                t * 5 - 2,
            )
            intrinsics = interpolate_intrinsics(
                batch["context"]["intrinsics"][0, 0],
                (
                    batch["context"]["intrinsics"][0, 1]
                    if v == 2
                    else batch["target"]["intrinsics"][0, 0]
                ),
                t * 5 - 2,
            )
            return extrinsics @ tf, intrinsics[None]

        return self.render_video_generic(
            batch,
            trajectory_fn,
            "interpolation_exagerrated",
            num_frames=300,
            smooth=False,
            loop_reverse=False,
        )

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
        gaussians_prob = self.encoder(batch["context"], self.global_step, False)
        gaussians_det = self.encoder(batch["context"], self.global_step, True)

        t = torch.linspace(0, 1, num_frames, dtype=torch.float32, device=self.device)
        if smooth:
            t = (torch.cos(torch.pi * (t + 1)) + 1) / 2

        extrinsics, intrinsics = trajectory_fn(t)

        _, _, _, h, w = batch["context"]["image"].shape

        # Color-map the result.
        def depth_map(result):
            near = result[result > 0][:16_000_000].quantile(0.01).log()
            far = result.view(-1)[:16_000_000].quantile(0.99).log()
            result = result.log()
            result = 1 - (result - near) / (far - near)
            return apply_color_map_to_image(result, "turbo")

        # TODO: Interpolate near and far planes?
        near = repeat(batch["context"]["near"][:, 0], "b -> b v", v=num_frames)
        far = repeat(batch["context"]["far"][:, 0], "b -> b v", v=num_frames)

        output_prob = self.decoder.forward(
            gaussians_prob, extrinsics, intrinsics, near, far, (h, w), "depth"
        )
        images_prob = [
            vcat(rgb, depth)
            for rgb, depth in zip(output_prob.color[0], depth_map(output_prob.depth[0]))
        ]
        output_det = self.decoder.forward(
            gaussians_det, extrinsics, intrinsics, near, far, (h, w), "depth"
        )
        images_det = [
            vcat(rgb, depth)
            for rgb, depth in zip(output_det.color[0], depth_map(output_det.depth[0]))
        ]
        images = [
            add_border(
                hcat(
                    add_label(image_prob, "Probabilistic"),
                    add_label(image_det, "Deterministic"),
                )
            )
            for image_prob, image_det in zip(images_prob, images_det)
        ]


        ############################################
        # video = torch.stack(images)
        # video = (video.clip(min=0, max=1) * 255).type(torch.uint8).cpu().numpy()
        # if loop_reverse:
        #     video = pack([video, video[::-1][1:-1]], "* c h w")[0]
        # visualizations = {
        #     f"video/{name}": wandb.Video(video[None], fps=30, format="mp4")
        # }


        # # Since the PyTorch Lightning doesn't support video logging, log to wandb directly.
        # try:
        #     wandb.log(visualizations)
        # except Exception:
        #     assert isinstance(self.logger, LocalLogger)
        #     for key, value in visualizations.items():
        #         tensor = value._prepare_video(value.data)
        #         clip = mpy.ImageSequenceClip(list(tensor), fps=value._fps)
        #         dir = LOG_PATH / key
        #         dir.mkdir(exist_ok=True, parents=True)
        #         clip.write_videofile(
        #             str(dir / f"{self.global_step:0>6}.mp4"), logger=None
        #         )
    

    def configure_optimizers(self):
        # Filter the parameters based on `requires_grad`.
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=self.optimizer_cfg.lr)
        
        warm_up_steps = self.optimizer_cfg.warm_up_steps
        warm_up = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            1 / warm_up_steps,
            1,
            total_iters=warm_up_steps,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": warm_up,
                "interval": "step",
                "frequency": 1,
            },
        }
