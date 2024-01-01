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
Implementation of mip-NeRF.
"""
from __future__ import annotations

import types
from typing import Type, Literal, Dict, List
from dataclasses import dataclass, field
from torch.nn import Parameter
from nerfstudio.engine.callbacks import TrainingCallback, TrainingCallbackAttributes, TrainingCallbackLocation
from nerfstudio.cameras.rays import RayBundle, RaySamples
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.models.vanilla_nerf import VanillaModelConfig
from nerfstudio.models.mipnerf import MipNerfModel
from nerfstudio.utils import colors
from linear.model_components.renderers import LinearRGBRenderer
from linear.model_components.ray_samplers import LinearPDFSampler, LinearUniformSampler
from linear.utils import get_weights_linear, get_density_linear
from nerfstudio.model_components.ray_samplers import UniformSampler, PDFSampler
from nerfstudio.fields.vanilla_nerf_field import NeRFField
from nerfstudio.field_components.encodings import NeRFEncoding


@dataclass
class LinearMipNerfModelConfig(VanillaModelConfig):
    """Configuration for model instantiation"""

    _target: Type = field(default_factory=lambda: LinearMipNerfModel)
    """target class to instantiate"""
    color_mode : Literal["midpoint", "left"] = "midpoint"
    farcolorfix: bool = False
    include_original: bool = False
    use_same_field: bool = True
    coarse_weight_anneal: bool = False 
    coarse_weight_anneal_end_iteration: int = 100000
    reverse_anneal: bool = False 
    num_constant_iterations: int = -1
    concat_walls: bool = False
    use_constant_importance_sampling: bool = False
    use_constant_aggregate_for_fine: bool = False
    



class LinearMipNerfModel(MipNerfModel):
    """mip-NeRF model

    Args:
        config: MipNerf configuration to instantiate model
    """

    config: LinearMipNerfModelConfig

    def __init__(
        self,
        config: LinearMipNerfModelConfig,
        **kwargs,
    ) -> None:
        if config.num_importance_samples <= 0:
            config.loss_coefficients["rgb_loss_fine"] = 0.0
            config.loss_coefficients["rgb_loss_coarse"] = 1.0
        
        super().__init__(config=config, **kwargs)
        self.init_coarse_weight = config.loss_coefficients["rgb_loss_coarse"]
        self.use_constant = config.num_constant_iterations > 0

    def populate_modules(self):
        """Set the fields and modules"""
        super().populate_modules()
        
        # fields
        position_encoding = NeRFEncoding(
            in_dim=3, num_frequencies=16, min_freq_exp=0.0, max_freq_exp=16.0, include_input=True
        )
        # position_encoding.pytorch_fwd = types.MethodType(pytorch_fwd_unscaled, position_encoding)
        
        
        direction_encoding = NeRFEncoding(
            in_dim=3, num_frequencies=4, min_freq_exp=0.0, max_freq_exp=4.0, include_input=True
        )
        # direction_encoding.pytorch_fwd = types.MethodType(pytorch_fwd_unscaled, direction_encoding)
        
        # samplers
        self.sampler_uniform = UniformSampler(num_samples=self.config.num_coarse_samples)
        if self.config.num_importance_samples > 0:
            if not self.config.use_same_field:
                self.fine_field = NeRFField(
                    position_encoding=position_encoding, direction_encoding=direction_encoding, use_integrated_encoding=True
                )
                # self.fine_field.get_density = types.MethodType(get_density_linear, self.fine_field)
            self.linear_sampler_pdf = LinearPDFSampler(num_samples=self.config.num_importance_samples, include_original=self.config.include_original, concat_walls=self.config.concat_walls)
            self.sampler_pdf = PDFSampler(num_samples=self.config.num_importance_samples, include_original=self.config.include_original)

        # renderers
        self.linear_renderer_rgb = LinearRGBRenderer(background_color=colors.WHITE, color_mode=self.config.color_mode, farcolorfix=self.config.farcolorfix, concat_walls=self.config.concat_walls)

    def get_training_callbacks(self, training_callback_attributes: TrainingCallbackAttributes) -> List[TrainingCallback]:
        cbs =  super().get_training_callbacks(training_callback_attributes)
        if self.config.coarse_weight_anneal:
            def set_coarse_weight(step):
                final_coarse_weight = 0.1 if self.config.reverse_anneal else 1.0
                slope = (final_coarse_weight - self.init_coarse_weight) / self.config.coarse_weight_anneal_end_iteration
                self.config.loss_coefficients["rgb_loss_coarse"] = self.init_coarse_weight + slope * min(step, self.config.coarse_weight_anneal_end_iteration)
            weight_cb = TrainingCallback(where_to_run=[TrainingCallbackLocation.BEFORE_TRAIN_ITERATION], update_every_num_iters=1, func=set_coarse_weight)
            cbs.append(weight_cb)
            
        if self.config.num_constant_iterations > 0:
            def toggle_constant(step):
                self.use_constant = step < self.config.num_constant_iterations
            toggle_constant_cb = TrainingCallback(where_to_run=[TrainingCallbackLocation.BEFORE_TRAIN_ITERATION], update_every_num_iters=1, func=toggle_constant)
            cbs.append(toggle_constant_cb)
        
        
        return cbs
    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        param_groups = super().get_param_groups()
        if not self.config.use_same_field:
            param_groups['fields'].extend(list(self.fine_field.parameters()))
        return param_groups

    def get_outputs(self, ray_bundle: RayBundle, **kwargs):
        if self.field is None:
            raise ValueError("populate_fields() must be called before get_outputs")

        # uniform sampling
        ray_samples_uniform: RaySamples
        ray_samples_uniform = self.sampler_uniform(ray_bundle)
        # First pass:
        field_outputs_coarse = self.field.forward(ray_samples_uniform)
        if not self.use_constant:
            mid_points = (ray_samples_uniform.frustums.ends + ray_samples_uniform.frustums.starts) / 2.0
            spacing_mid_points = (ray_samples_uniform.spacing_ends + ray_samples_uniform.spacing_starts) / 2.0
            ray_samples_uniform_shifted = ray_bundle.get_ray_samples(
                bin_starts=mid_points[:, :-1], 
                bin_ends=mid_points[:, 1:], 
                spacing_starts=spacing_mid_points[:, :-1], 
                spacing_ends=spacing_mid_points[:, 1:], 
                spacing_to_euclidean_fn=ray_samples_uniform.spacing_to_euclidean_fn)
            ray_samples_uniform = ray_samples_uniform_shifted
            
            weights_coarse, densities_coarse, transmittance_coarse = get_weights_linear(ray_samples_uniform, field_outputs_coarse[FieldHeadNames.DENSITY], concat_walls=self.config.concat_walls)
            rgb_coarse = self.linear_renderer_rgb(
                rgb=field_outputs_coarse[FieldHeadNames.RGB],
                weights=weights_coarse,
            )
        else:
            weights_coarse = ray_samples_uniform.get_weights(field_outputs_coarse[FieldHeadNames.DENSITY])
            rgb_coarse = self.renderer_rgb(
                rgb=field_outputs_coarse[FieldHeadNames.RGB],
                weights=weights_coarse,
            )
            
        accumulation_coarse = self.renderer_accumulation(weights_coarse)
        depth_coarse = self.renderer_depth(weights_coarse, ray_samples_uniform)
            
        
        # pdf sampling
        if self.config.num_importance_samples > 0:
            if not (self.use_constant or self.config.use_constant_importance_sampling):
                ray_samples_pdf = self.linear_sampler_pdf(ray_bundle, ray_samples_uniform, weights_coarse, densities_coarse, transmittance_coarse)
            else:
                ray_samples_pdf = self.sampler_pdf(ray_bundle, ray_samples_uniform, weights_coarse)

            # Second pass:
            if not self.config.use_same_field:
                field_outputs_fine = self.fine_field.forward(ray_samples_pdf)
            else:
                field_outputs_fine = self.field.forward(ray_samples_pdf)
                
            if not (self.use_constant or self.config.use_constant_aggregate_for_fine):
                mid_points = (ray_samples_pdf.frustums.ends + ray_samples_pdf.frustums.starts) / 2.0
                spacing_mid_points = (ray_samples_pdf.spacing_ends + ray_samples_pdf.spacing_starts) / 2.0
                pdf_ray_samples_shifted = ray_bundle.get_ray_samples(
                    bin_starts=mid_points[:, :-1],
                    bin_ends=mid_points[:, 1:],
                    spacing_starts=spacing_mid_points[:, :-1],
                    spacing_ends=spacing_mid_points[:, 1:], 
                    spacing_to_euclidean_fn=ray_samples_pdf.spacing_to_euclidean_fn)
                ray_samples_pdf = pdf_ray_samples_shifted
                weights_fine, _, _ = get_weights_linear(ray_samples_pdf, field_outputs_fine[FieldHeadNames.DENSITY], concat_walls=self.config.concat_walls)
                
                rgb_fine = self.linear_renderer_rgb(
                    rgb=field_outputs_fine[FieldHeadNames.RGB],
                    weights=weights_fine,
                )
            else:
                weights_fine = ray_samples_pdf.get_weights(field_outputs_fine[FieldHeadNames.DENSITY])
                rgb_fine = self.renderer_rgb(
                    rgb=field_outputs_fine[FieldHeadNames.RGB],
                    weights=weights_fine,
                )
            accumulation_fine = self.renderer_accumulation(weights_fine)
            depth_fine = self.renderer_depth(weights_fine, ray_samples_pdf)
        else:
            rgb_fine = rgb_coarse
            accumulation_fine = accumulation_coarse
            depth_fine = depth_coarse
            
        outputs = {
            "rgb_coarse": rgb_coarse,
            "rgb_fine": rgb_fine,
            "accumulation_coarse": accumulation_coarse,
            "accumulation_fine": accumulation_fine,
            "depth_coarse": depth_coarse,
            "depth_fine": depth_fine,
        }
        return outputs