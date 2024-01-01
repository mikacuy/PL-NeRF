# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Implementation of vanilla nerf.
"""

from __future__ import annotations
import types
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple, Type

import torch
from torch.nn import Parameter
from torchmetrics import PeakSignalNoiseRatio
from torchmetrics.functional import structural_similarity_index_measure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from nerfstudio.cameras.rays import RayBundle
from nerfstudio.configs.config_utils import to_immutable_dict
from nerfstudio.field_components.encodings import NeRFEncoding
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.field_components.temporal_distortions import TemporalDistortionKind
from nerfstudio.fields.vanilla_nerf_field import NeRFField
from nerfstudio.model_components.losses import MSELoss
from nerfstudio.model_components.ray_samplers import PDFSampler, UniformSampler
from nerfstudio.model_components.renderers import (
    AccumulationRenderer,
    DepthRenderer,
    RGBRenderer,
)
from nerfstudio.models.base_model import Model, ModelConfig
from nerfstudio.models.vanilla_nerf import VanillaModelConfig, NeRFModel
from nerfstudio.utils import colormaps, colors, misc
from nerfstudio.model_components.scene_colliders import NearFarCollider
from linear.model_components.ray_samplers import LinearPDFSampler, LinearProposalNetworkSampler
from linear.model_components.renderers import LinearRGBRenderer
from linear.utils import get_weights_linear, get_density_linear, get_outputs_linear


@dataclass
class LinearVanillaModelConfig(VanillaModelConfig):
    """Vanilla Model Config"""

    _target: Type = field(default_factory=lambda: LinearNeRFModel)
    num_coarse_samples: int = 128
    """Number of samples in coarse field evaluation"""
    num_importance_samples: int = 64
    """Number of samples in fine field evaluation"""



class LinearNeRFModel(NeRFModel):
    """Vanilla NeRF model

    Args:
        config: Basic NeRF configuration to instantiate model
    """

    config: LinearVanillaModelConfig


    def populate_modules(self):
        """Set the fields and modules"""
        super().populate_modules()


        # samplers
        self.sampler_uniform = UniformSampler(num_samples=self.config.num_coarse_samples)
        self.sampler_pdf = LinearPDFSampler(num_samples=self.config.num_importance_samples)

        # renderers
        self.renderer_rgb = LinearRGBRenderer(background_color=colors.WHITE)
        
        self.field_coarse.get_density = types.MethodType(get_density_linear, self.field_coarse)
        self.field_coarse.get_outputs = types.MethodType(get_outputs_linear, self.field_coarse)
        self.field_fine.get_density = types.MethodType(get_density_linear, self.field_fine)
        self.field_fine.get_outputs = types.MethodType(get_outputs_linear, self.field_fine)

    def get_outputs(self, ray_bundle: RayBundle, **kwargs):
        if self.field_coarse is None or self.field_fine is None:
            raise ValueError("populate_fields() must be called before get_outputs")

        # uniform sampling
        ray_samples_uniform = self.sampler_uniform(ray_bundle)
        if self.temporal_distortion is not None:
            offsets = None
            if ray_samples_uniform.times is not None:
                offsets = self.temporal_distortion(
                    ray_samples_uniform.frustums.get_positions(), ray_samples_uniform.times
                )
            ray_samples_uniform.frustums.set_offsets(offsets)

        # coarse field:
        field_outputs_coarse = self.field_coarse.forward(ray_samples_uniform)
        weights_coarse, densities_coarse, transmittance_coarse = get_weights_linear(ray_samples_uniform, field_outputs_coarse[FieldHeadNames.DENSITY])
        rgb_coarse = self.renderer_rgb(
            rgb=field_outputs_coarse[FieldHeadNames.RGB],
            weights=weights_coarse,
        )
        accumulation_coarse = self.renderer_accumulation(weights_coarse)
        depth_coarse = self.renderer_depth(weights_coarse, ray_samples_uniform)

        # pdf sampling
        print(ray_bundle.shape, ray_samples_uniform.shape, weights_coarse.shape, densities_coarse.shape, transmittance_coarse.shape)
        ray_samples_pdf = self.sampler_pdf(ray_bundle, ray_samples_uniform, weights_coarse, densities_coarse, transmittance_coarse)
        print(ray_samples_pdf.shape)
        if self.temporal_distortion is not None:
            offsets = None
            if ray_samples_pdf.times is not None:
                offsets = self.temporal_distortion(ray_samples_pdf.frustums.get_positions(), ray_samples_pdf.times)
            ray_samples_pdf.frustums.set_offsets(offsets)

        # fine field:
        field_outputs_fine = self.field_fine.forward(ray_samples_pdf)
        weights_fine, _, _ = get_weights_linear(ray_samples_pdf, field_outputs_fine[FieldHeadNames.DENSITY])
        rgb_fine = self.renderer_rgb(
            rgb=field_outputs_fine[FieldHeadNames.RGB],
            weights=weights_fine,
        )
        accumulation_fine = self.renderer_accumulation(weights_fine)
        depth_fine = self.renderer_depth(weights_fine, ray_samples_pdf)

        outputs = {
            "rgb_coarse": rgb_coarse,
            "rgb_fine": rgb_fine,
            "accumulation_coarse": accumulation_coarse,
            "accumulation_fine": accumulation_fine,
            "depth_coarse": depth_coarse,
            "depth_fine": depth_fine,
        }
        return outputs
