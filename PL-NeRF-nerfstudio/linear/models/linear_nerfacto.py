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
NeRF implementation that combines many recent advancements.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Tuple, Type

import numpy as np
import torch
from torch.nn import Parameter
from torchmetrics import PeakSignalNoiseRatio
from torchmetrics.functional import structural_similarity_index_measure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from nerfstudio.cameras.rays import RayBundle, RaySamples
from nerfstudio.engine.callbacks import TrainingCallback, TrainingCallbackAttributes, TrainingCallbackLocation
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.field_components.spatial_distortions import SceneContraction
from nerfstudio.fields.density_fields import HashMLPDensityField
from nerfstudio.fields.nerfacto_field import NerfactoField
from nerfstudio.model_components.losses import (
    MSELoss,
    distortion_loss,
    interlevel_loss,
    orientation_loss,
    pred_normal_loss,
    scale_gradients_by_distance_squared,
)
from nerfstudio.model_components.ray_samplers import ProposalNetworkSampler, UniformSampler
from nerfstudio.model_components.renderers import AccumulationRenderer, DepthRenderer, NormalsRenderer, RGBRenderer
from nerfstudio.model_components.scene_colliders import NearFarCollider, AABBBoxCollider
from nerfstudio.model_components.shaders import NormalsShader
from nerfstudio.models.base_model import Model, ModelConfig
from nerfstudio.models.nerfacto import NerfactoModelConfig, NerfactoModel
from nerfstudio.utils import colormaps
from linear.model_components.renderers import LinearRGBRenderer
from linear.model_components.ray_samplers import LinearPDFSampler, LinearUniformSampler, LinearProposalNetworkSampler
from linear.fields.linear_nerfacto_field import LinearNerfactoField

@dataclass
class LinearNerfactoModelConfig(NerfactoModelConfig):
    """Configuration for model instantiation"""

    _target: Type = field(default_factory=lambda: LinearNerfactoModel)
    """target class to instantiate"""
    color_mode : Literal["midpoint", "left"] = "midpoint"
    farcolorfix: bool = False
    num_constant_iterations: int = -1
    concat_walls: bool = True
    include_original: bool = False
    



class LinearNerfactoModel(NerfactoModel):
    """Nerfacto model

    Args:
        config: Nerfacto configuration to instantiate model
    """

    config: LinearNerfactoModelConfig

    def populate_modules(self):
        """Set the fields and modules."""
        super().populate_modules()
        
        # Samplers
        def update_schedule(step):
            return np.clip(
                np.interp(step, [0, self.config.proposal_warmup], [0, self.config.proposal_update_every]),
                1,
                self.config.proposal_update_every,
            )

        # Change proposal network initial sampler if uniform
        initial_sampler = None  # None is for piecewise as default (see ProposalNetworkSampler)
        if self.config.proposal_initial_sampler == "uniform":
            initial_sampler = UniformSampler(single_jitter=self.config.use_single_jitter)

        self.linear_proposal_sampler = ProposalNetworkSampler(
            num_nerf_samples_per_ray=self.config.num_nerf_samples_per_ray,
            num_proposal_samples_per_ray=self.config.num_proposal_samples_per_ray,
            num_proposal_network_iterations=self.config.num_proposal_iterations,
            single_jitter=self.config.use_single_jitter,
            update_sched=update_schedule,
            initial_sampler=initial_sampler,
            include_original=self.config.include_original
        )

        # renderers
        self.linear_renderer_rgb = LinearRGBRenderer(background_color=self.config.background_color, concat_walls=self.config.concat_walls)
        self.use_constant = self.config.num_constant_iterations > 0

        print(self)

    
    def get_training_callbacks(
        self, training_callback_attributes: TrainingCallbackAttributes
    ) -> List[TrainingCallback]:
        callbacks = super().get_training_callbacks(training_callback_attributes=training_callback_attributes)
        if self.config.num_constant_iterations > 0:
            def toggle_constant(step):
                self.use_constant = step < self.config.num_constant_iterations
            toggle_constant_cb = TrainingCallback(where_to_run=[TrainingCallbackLocation.BEFORE_TRAIN_ITERATION], update_every_num_iters=1, func=toggle_constant)
            callbacks.append(toggle_constant_cb)
        
        return callbacks

    def get_outputs(self, ray_bundle: RayBundle, return_samples:bool=False, **kwargs):
        ray_samples: RaySamples
        if self.use_constant:
            ray_samples, weights_list, ray_samples_list = self.proposal_sampler(ray_bundle, density_fns=self.density_fns)
        else:
            ray_samples, weights_list, ray_samples_list = self.linear_proposal_sampler(ray_bundle, density_fns=self.density_fns)

        ray_samples.frustums.sample_method = "piecewise_linear"
        field_outputs = self.field.forward(ray_samples, compute_normals=self.config.predict_normals)
        if self.config.use_gradient_scaling:
            field_outputs = scale_gradients_by_distance_squared(field_outputs, ray_samples)
            
        if self.use_constant:
            weights = ray_samples.get_weights(field_outputs[FieldHeadNames.DENSITY])
            weights_list.append(weights)
            ray_samples_list.append(ray_samples)
        else:
            weights, _, _ = ray_samples.get_weights_linear(field_outputs[FieldHeadNames.DENSITY], concat_walls=self.config.concat_walls)
            if self.config.concat_walls:
                weights_list.append(weights[..., 1:-1, :])
            else:
                weights_list.append(weights)
            ray_samples_list.append(ray_samples)

        if self.use_constant:
            rgb = self.renderer_rgb(rgb=field_outputs[FieldHeadNames.RGB], weights=weights)
        else:
            rgb = self.linear_renderer_rgb(rgb=field_outputs[FieldHeadNames.RGB], weights=weights)
            
        depth = self.renderer_depth(weights=weights, ray_samples=ray_samples)
        accumulation = self.renderer_accumulation(weights=weights)

        outputs = {
            "rgb": rgb,
            "accumulation": accumulation,
            "depth": depth,
        }

        if self.config.predict_normals:
            normals = self.renderer_normals(normals=field_outputs[FieldHeadNames.NORMALS], weights=weights)
            pred_normals = self.renderer_normals(field_outputs[FieldHeadNames.PRED_NORMALS], weights=weights)
            outputs["normals"] = self.normals_shader(normals)
            outputs["pred_normals"] = self.normals_shader(pred_normals)
        # These use a lot of GPU memory, so we avoid storing them for eval.
        if self.training:
            outputs["weights_list"] = weights_list
            outputs["ray_samples_list"] = ray_samples_list

        if self.training and self.config.predict_normals:
            outputs["rendered_orientation_loss"] = orientation_loss(
                weights.detach(), field_outputs[FieldHeadNames.NORMALS], ray_bundle.directions
            )

            outputs["rendered_pred_normal_loss"] = pred_normal_loss(
                weights.detach(),
                field_outputs[FieldHeadNames.NORMALS].detach(),
                field_outputs[FieldHeadNames.PRED_NORMALS],
            )

        for i in range(self.config.num_proposal_iterations):
            outputs[f"prop_depth_{i}"] = self.renderer_depth(weights=weights_list[i], ray_samples=ray_samples_list[i])
        return outputs