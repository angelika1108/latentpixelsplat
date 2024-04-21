from .encoder_latent import EncoderLatent, EncoderLatentTiny
from .visualization.encoder_visualizer_epipolar_cfg import EncoderVisualizerEpipolarCfg
from .epipolar.epipolar_transformer import EpipolarTransformer, EpipolarTransformerCfg
from .epipolar.depth_predictor_monocular import DepthPredictorMonocular
from .encoder import Encoder
from .common.gaussian_adapter import GaussianAdapter, GaussianAdapterCfg
from .backbone import Backbone, BackboneCfg, get_backbone
from ..types import Gaussians
from ...geometry.projection import sample_image_grid
from ...dataset.types import BatchedExample, DataShim
from ...dataset.shims.patch_shim import apply_patch_shim
from ...dataset.shims.bounds_shim import apply_bounds_shim
import yaml
from torch.nn import functional as F
from torch import Tensor, nn
from jaxtyping import Float
from einops import rearrange
import torch
from dataclasses import dataclass
from typing import Literal, Optional
import time
import sys
sys.path.append('../../../')


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
    encoder_latent_type: str | None
    d_latent: int
    gaussian_grid_size: list[int]


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

        self.encoder_latent_type = cfg.encoder_latent_type  # "medium" or "tiny" or None
        self.d_latent = cfg.d_latent

        if self.encoder_latent_type is None:
            self.encoder_latent = None
            # self.d_latent = 3

        elif self.encoder_latent_type == "medium":
            config_path = "config/model/encoder/latent_medium/config_vq-f4-noattn.yaml"
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)
            self.encoder_latent = EncoderLatent(
                **config['model']['params']['ddconfig'], **config['model']['params'])
            # self.d_latent = config['model']['params']['embed_dim']

        elif self.encoder_latent_type == "tiny":
            self.encoder_latent = EncoderLatentTiny(
                d_in=3, d_out=self.d_latent, downsample=4)

        else:
            raise ValueError(
                f"Unknown encoder_latent_type: {self.encoder_latent_type}")

        if self.encoder_latent is not None:
            self.downsample = self.encoder_latent.downsample

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
        self.gaussian_adapter = GaussianAdapter(
            cfg.gaussian_adapter, self.d_latent)

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

        if self.d_latent in [4, 5, 6]:
            self.to_gaussians_2 = nn.Sequential(
                nn.ReLU(),
                nn.Linear(
                    cfg.d_feature,
                    cfg.num_surfaces * (self.gaussian_adapter.d_in - 7), # don't predict the first 7 parameters (scales and rotations) and the offset_xy, in total 9 parameters
                ),
            )

        self.high_resolution_skip = nn.Sequential(
            nn.Conv2d(self.d_latent, cfg.d_feature, 7, 1, 3),
            nn.ReLU(),
        )

        # self.feature_downscaler = nn.Conv2d(cfg.d_feature, cfg.d_feature, 2, 2) if self.downsample == 8 else None

    def map_pdf_to_opacity(
        self,
        pdf: Float[Tensor, " *batch"],
        global_step: int,
    ) -> Float[Tensor, " *batch"]:
        # https://www.desmos.com/calculator/opvwti3ba9

        # Figure out the exponent.
        cfg = self.cfg.opacity_mapping
        x = cfg.initial + min(global_step / cfg.warm_up,
                              1) * (cfg.final - cfg.initial)
        exponent = 2**x

        # Map the probability density to an opacity.
        return 0.5 * (1 - (1 - pdf) ** exponent + pdf ** (1 / exponent))

    def forward(
        self,
        context: dict,
        global_step: int,
        deterministic: bool = False,
        visualization_dump: Optional[dict] = None):

        # torch.cuda.synchronize()
        # t0 = time.time()

        device = context["image"].device
        b, v, _, h, w = context["image"].shape

        # Encode the context images.
        features = self.backbone(context)
        features = rearrange(features, "b v c h w -> b v h w c")
        features = self.backbone_projection(features)
        features = rearrange(features, "b v h w c -> b v c h w")

        # torch.cuda.synchronize()
        # t_backbone = time.time() - t0
        # torch.cuda.synchronize()
        # t0 = time.time()

        # Run the epipolar transformer.
        if self.cfg.use_epipolar_transformer:
            features, sampling = self.epipolar_transformer(
                features,
                context["extrinsics"],
                context["intrinsics"],
                context["near"],
                context["far"],
            )

        assert h // features.shape[3] == w // features.shape[4]
        dowsample_factor = h // features.shape[3]

        # torch.cuda.synchronize()
        # t_epipolar_transformer = time.time() - t0
        # torch.cuda.synchronize()
        # t0 = time.time()
        
        # Add the high-resolution skip connection.
        skip = rearrange(context["image"], "b v c h w -> (b v) c h w")

        # Input channels: 3, output channels: 3
        if self.encoder_latent is not None:
            # Calculate latent skip connection.
            if isinstance(self.encoder_latent, EncoderLatent):
                # Input channels: 3, output channels: 3
                skip = self.encoder_latent(skip)
            elif isinstance(self.encoder_latent, EncoderLatentTiny):
                # Input channels: 3, output channels: d_latent
                skip = self.encoder_latent(skip)
            else:
                raise ValueError("Unknown latent encoder type")

        # torch.cuda.synchronize()
        # t_latent_encoder = time.time() - t0
        # torch.cuda.synchronize()
        # t0 = time.time()

        skip = self.high_resolution_skip(skip)
        
        if self.encoder_latent is None:
            skip = F.interpolate(skip, size=(features.shape[3], features.shape[4]), mode="bilinear", align_corners=True)
        
        features = features + \
            rearrange(skip, "(b v) c h w -> b v c h w", b=b, v=v)

        # Sample depths from the resulting features.
        features = rearrange(features, "b v c h w -> b v (h w) c")
        depths, densities = self.depth_predictor.forward(
            features,
            context["near"],
            context["far"],
            deterministic,
            1 if deterministic else self.cfg.gaussians_per_pixel,
        )

        # torch.cuda.synchronize()
        # t_depth = time.time() - t0
        # torch.cuda.synchronize()
        # t0 = time.time()

        # Convert the features and depths into Gaussians.

        if self.encoder_latent_type is not None:
            h_down = h // self.downsample
            w_down = w // self.downsample
        else:
            h_down = h // dowsample_factor
            w_down = w // dowsample_factor
        
        # breakpoint()
        xy_ray, _ = sample_image_grid((h_down, w_down), device)
        xy_ray = rearrange(xy_ray, "h w xy -> (h w) () xy")
        gaussians_1 = rearrange(
            self.to_gaussians(features),
            "... (srf c) -> ... srf c",
            srf=self.cfg.num_surfaces,
        )

        offset_xy = gaussians_1[..., :2].sigmoid()
        pixel_size = 1 / \
            torch.tensor((w_down, h_down), dtype=torch.float32, device=device)
        xy_ray = xy_ray + (offset_xy - 0.5) * pixel_size
        gpp = self.cfg.gaussians_per_pixel

        gaussians = self.gaussian_adapter.forward(
            rearrange(context["extrinsics"], "b v i j -> b v () () () i j"),
            rearrange(context["intrinsics"], "b v i j -> b v () () () i j"),
            rearrange(xy_ray, "b v r srf xy -> b v r srf () xy"),
            depths,
            self.map_pdf_to_opacity(densities, global_step) / gpp,
            rearrange(gaussians_1[..., 2:], "b v r srf c -> b v r srf () c"),
            (self.cfg.gaussian_grid_size[0], self.cfg.gaussian_grid_size[1]),  # or (h_down, w_down)
        )

        #######################################################################
        if self.d_latent in [4, 5, 6]:
            gaussians_2 = rearrange(
                self.to_gaussians_2(features),
                "... (srf c) -> ... srf c",
                srf=self.cfg.num_surfaces,
            )
            gaussians_2 = torch.cat((gaussians_1[..., 2:9], gaussians_2), dim=-1)

            gaussians_2 = self.gaussian_adapter.forward(
                rearrange(context["extrinsics"], "b v i j -> b v () () () i j"),
                rearrange(context["intrinsics"], "b v i j -> b v () () () i j"),
                rearrange(xy_ray, "b v r srf xy -> b v r srf () xy"),
                depths,
                self.map_pdf_to_opacity(densities, global_step) / gpp,
                rearrange(gaussians_2, "b v r srf c -> b v r srf () c"),
                (self.cfg.gaussian_grid_size[0], self.cfg.gaussian_grid_size[1]),  # or (h_down, w_down)
            )


        # torch.cuda.synchronize()
        # t_gaussian_adapter = time.time() - t0
        # torch.cuda.synchronize()
        # t0 = time.time()
        # breakpoint()

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

        if self.d_latent == 3:
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
        elif self.d_latent in [4, 5, 6]:
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
            ), Gaussians(
                    rearrange(
                        gaussians.means,
                        "b v r srf spp xyz -> b (v r srf spp) xyz",
                    ),
                    rearrange(
                        gaussians.covariances,
                        "b v r srf spp i j -> b (v r srf spp) i j",
                    ),
                    rearrange(
                        gaussians_2.harmonics,
                        "b v r srf spp c d_sh -> b (v r srf spp) c d_sh",
                    ),
                    rearrange(
                        opacity_multiplier * gaussians.opacities,
                        "b v r srf spp -> b (v r srf spp)",
                    ),
                )
        else:
           raise ValueError(f"Invalid d_latent: {self.d_latent}")             
        

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
