from dataclasses import dataclass
from typing import Literal, Optional
import time 

import torch
from einops import rearrange
from jaxtyping import Float
from torch import Tensor, nn
from torch.profiler import profile, record_function, ProfilerActivity
from lightning.pytorch.profilers import SimpleProfiler, PassThroughProfiler
from torch.nn import functional as F
import yaml

from ...dataset.shims.bounds_shim import apply_bounds_shim
from ...dataset.shims.patch_shim import apply_patch_shim
from ...dataset.types import BatchedExample, DataShim
from ...geometry.projection import sample_image_grid
from ..types import Gaussians
from .backbone import Backbone, BackboneCfg, get_backbone
from .common.gaussian_adapter import GaussianAdapter, GaussianAdapterCfg
from .encoder import Encoder
from .epipolar.depth_predictor_monocular import DepthPredictorMonocular
from .epipolar.epipolar_transformer import EpipolarTransformer, EpipolarTransformerCfg
from .visualization.encoder_visualizer_epipolar_cfg import EncoderVisualizerEpipolarCfg
from .encoder_latent import EncoderLatent, EncoderLatentTiny

@dataclass
class OpacityMappingCfg:
    initial: float
    final: float
    warm_up: int


@dataclass
class EncoderEpipolarCfg:
    name: Literal["epipolar"]
    d_feature: int
    num_monocular_samples: int
    num_surfaces: int
    predict_opacity: bool
    backbone: BackboneCfg
    visualizer: EncoderVisualizerEpipolarCfg
    near_disparity: float
    gaussian_adapter: GaussianAdapterCfg
    apply_bounds_shim: bool
    epipolar_transformer: EpipolarTransformerCfg
    opacity_mapping: OpacityMappingCfg
    gaussians_per_pixel: int
    use_epipolar_transformer: bool
    use_transmittance: bool


class EncoderEpipolar(Encoder[EncoderEpipolarCfg]):
    backbone: Backbone
    backbone_projection: nn.Sequential
    encoder_latent: nn.Module
    epipolar_transformer: EpipolarTransformer | None
    depth_predictor: DepthPredictorMonocular
    to_gaussians: nn.Sequential
    gaussian_adapter: GaussianAdapter
    high_resolution_skip: nn.Sequential

    def __init__(self, cfg: EncoderEpipolarCfg) -> None:
        super().__init__(cfg)

        self.backbone = get_backbone(cfg.backbone, 3)
        self.backbone_projection = nn.Sequential(
            nn.ReLU(),
            nn.Linear(self.backbone.d_out, cfg.d_feature),
        )
        
        ########################################################################################

        # self.profiler = PassThroughProfiler(dirpath="/home/angelika/pixelsplat/chrome_traces", 
        #                                     filename="profile_encoder.txt")
        self.profiler = SimpleProfiler(dirpath="/home/angelika/pixelsplat/chrome_traces", 
                                            filename="profile_encoder.txt")

        ########################################################################################

        config_path = "/home/angelika/vae/configs/config_vq-f4-noattn.yaml"
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        self.encoder_latent = EncoderLatent(**config['model']['params']['ddconfig'], **config['model']['params'])

        self.encoder_latent = EncoderLatentTiny()


        if cfg.use_epipolar_transformer:
            self.epipolar_transformer = EpipolarTransformer(
                cfg.epipolar_transformer,
                cfg.d_feature,
            )
        else:
            self.epipolar_transformer = None
        
        self.depth_predictor = DepthPredictorMonocular(
            cfg.d_feature,
            cfg.num_monocular_samples,
            cfg.num_surfaces,
            cfg.use_transmittance,
        )        
        self.gaussian_adapter = GaussianAdapter(cfg.gaussian_adapter)
        if cfg.predict_opacity:
            self.to_opacity = nn.Sequential(
                nn.ReLU(),
                nn.Linear(cfg.d_feature, 1),
                nn.Sigmoid(),
            )
        self.to_gaussians = nn.Sequential(
            nn.ReLU(),
            nn.Linear(
                cfg.d_feature,
                cfg.num_surfaces * (2 + self.gaussian_adapter.d_in),
            ),
        )
        self.high_resolution_skip = nn.Sequential(
            nn.Conv2d(3, cfg.d_feature, 7, 1, 3),
            nn.ReLU(),
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

    def forward(
        self,
        context: dict,
        global_step: int,
        deterministic: bool = False,
        visualization_dump: Optional[dict] = None,
    ) -> Gaussians:
        device = context["image"].device
        b, v, _, h, w = context["image"].shape        
        
        with self.profiler.profile("encoder"):
        # with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], 
                    #  with_stack=True, 
                    #  profile_memory=True, 
                    #  record_shapes=True) as prof_encoder:
            
            # with record_function("enc_backbone"):
            
            t0 = time.time()        

            # Encode the context images.
            features = self.backbone(context)
            features = rearrange(features, "b v c h w -> b v h w c")
            features = self.backbone_projection(features)
            features = rearrange(features, "b v h w c -> b v c h w")

            t_backbone = time.time() - t0
            t0 = time.time() 

            # with record_function("enc_epipolar_transformer"):

            # Run the epipolar transformer.
            if self.cfg.use_epipolar_transformer:
                features, sampling = self.epipolar_transformer(
                    features,
                    context["extrinsics"],
                    context["intrinsics"],
                    context["near"],
                    context["far"],
                )

            t_epipolar_transformer = time.time() - t0
            t0 = time.time()

            features = rearrange(features, "b v c h w -> (b v) c h w")
            features = F.interpolate(features, size=(h//4, w//4), mode="bilinear", align_corners=False)
            # features = F.avg_pool2d(features, kernel_size=4, stride=4)
            features = rearrange(features, "(b v) c h w -> b v c h w", b=b, v=v)

            # # Add the high-resolution skip connection.
            # skip = rearrange(context["image"], "b v c h w -> (b v) c h w")
            # skip = self.high_resolution_skip(skip)
            # features = features + rearrange(skip, "(b v) c h w -> b v c h w", b=b, v=v)

            ############################################################################
                
            # with record_function("enc_latent"):

            # Add latent skip connection.
            skip = rearrange(context["image"], "b v c h w -> (b v) c h w")

            if isinstance(self.encoder_latent, EncoderLatent):    # Input channels: 3, output channels: 3
                skip = self.encoder_latent(skip)
                skip = F.interpolate(skip, size=(h//4, w//4), mode="bilinear", align_corners=False)
            elif isinstance(self.encoder_latent, EncoderLatentTiny):    # Input channels: 3, output channels: 4
                skip = self.encoder_latent(skip)
                skip = skip[:, :-1, :, :]
            else:
                raise ValueError("Unknown latent encoder type")

            t_latent_encoder = time.time() - t0
            t0 = time.time()
            
            skip = self.high_resolution_skip(skip)
            features = features + rearrange(skip, "(b v) c h w -> b v c h w", b=b, v=v)
            
            # # Max pool for debug ##############################################
            # features = rearrange(features, "b v c h w -> (b v) c h w")
            # features = F.max_pool2d(features, kernel_size=4, stride=4)
            # features = rearrange(features, "(b v) c h w -> b v c h w", b=b, v=v)


            # with record_function("enc_sample_depths"):

            # Sample depths from the resulting features.
            features = rearrange(features, "b v c h w -> b v (h w) c")
            depths, densities = self.depth_predictor.forward(
                features,
                context["near"],
                context["far"],
                deterministic,
                1 if deterministic else self.cfg.gaussians_per_pixel,
            )

            # with record_function("enc_gaussian_adapter"):

            # Convert the features and depths into Gaussians.
            h_down = h // 4   ######################################################################
            w_down = w // 4
            
            xy_ray, _ = sample_image_grid((h_down, w_down), device)
            xy_ray = rearrange(xy_ray, "h w xy -> (h w) () xy")            
            gaussians = rearrange(
                self.to_gaussians(features),
                "... (srf c) -> ... srf c",
                srf=self.cfg.num_surfaces,
            )
            offset_xy = gaussians[..., :2].sigmoid()
            pixel_size = 1 / torch.tensor((w_down, h_down), dtype=torch.float32, device=device)

            xy_ray = xy_ray + (offset_xy - 0.5) * pixel_size
            gpp = self.cfg.gaussians_per_pixel
        
            gaussians = self.gaussian_adapter.forward(
                rearrange(context["extrinsics"], "b v i j -> b v () () () i j"),
                rearrange(context["intrinsics"], "b v i j -> b v () () () i j"),
                rearrange(xy_ray, "b v r srf xy -> b v r srf () xy"),
                depths,
                self.map_pdf_to_opacity(densities, global_step) / gpp,
                rearrange(gaussians[..., 2:], "b v r srf c -> b v r srf () c"),
                (h_down, w_down),
            )
        
            t_gaussian_adapter = time.time() - t0
            # breakpoint()

        # prof_encoder.export_chrome_trace("chrome_traces/trace_encoder.json")
        # print(prof_encoder.key_averages().table(sort_by="self_cuda_memory_usage", row_limit=30))

        # Dump visualizations if needed.
        if visualization_dump is not None:
            visualization_dump["depth"] = rearrange(
                depths, "b v (h w) srf s -> b v h w srf s", h=h_down, w=w_down
            )
            visualization_dump["scales"] = rearrange(
                gaussians.scales, "b v r srf spp xyz -> b (v r srf spp) xyz"
            )
            visualization_dump["rotations"] = rearrange(
                gaussians.rotations, "b v r srf spp xyzw -> b (v r srf spp) xyzw"
            )
            if self.cfg.use_epipolar_transformer:
                visualization_dump["sampling"] = sampling

        # Optionally apply a per-pixel opacity.
        opacity_multiplier = (
            rearrange(self.to_opacity(features), "b v r () -> b v r () ()")
            if self.cfg.predict_opacity
            else 1
        )

        return Gaussians(
            rearrange(
                gaussians.means,
                "b v r srf spp xyz -> b (v r srf spp) xyz",
            ),
            rearrange(
                gaussians.covariances,
                "b v r srf spp i j -> b (v r srf spp) i j",
            ),
            rearrange(
                gaussians.harmonics,
                "b v r srf spp c d_sh -> b (v r srf spp) c d_sh",
            ),
            rearrange(
                opacity_multiplier * gaussians.opacities,
                "b v r srf spp -> b (v r srf spp)",
            ),
        )

    def get_data_shim(self) -> DataShim:
        def data_shim(batch: BatchedExample) -> BatchedExample:
            batch = apply_patch_shim(
                batch,
                patch_size=self.cfg.epipolar_transformer.self_attention.patch_size
                * self.cfg.epipolar_transformer.downscale,
            )

            if self.cfg.apply_bounds_shim:
                _, _, _, h, w = batch["context"]["image"].shape
                near_disparity = self.cfg.near_disparity * min(h, w)
                batch = apply_bounds_shim(batch, near_disparity, 0.5)

            return batch

        return data_shim

    @property
    def sampler(self):
        # hack to make the visualizer work
        return self.epipolar_transformer.epipolar_sampler
