import torch 
from torch import Tensor 
from typing import Tuple, Optional, Dict
from jaxtyping import Float
import torch.nn.functional as F
from nerfstudio.cameras.rays import RaySamples
from nerfstudio.field_components.field_heads import (
    FieldHeadNames,
)
from nerfstudio.utils.math import expected_sin
def get_weights_linear(ray_samples: RaySamples, densities: Float[Tensor, "*batch num_samples 1"], concat_walls: bool=True) -> Float[Tensor, "*batch num_samples 1"]:
    deltas = ray_samples.deltas
    interval_ave_densities = 0.5 * (densities[..., 1:, :] + densities[..., :-1, :]) # N + 2 (N)
    delta_density = deltas * interval_ave_densities # N + 2 (N)
    alphas = 1 - torch.exp(-delta_density)
    transmittance = torch.cumsum(delta_density[..., :-1, :], dim=-2) # N + 1 (N - 1)
    transmittance = torch.cat(
        [torch.zeros((*transmittance.shape[:1], 1, 1), device=densities.device), transmittance], dim=-2
    ) # N + 2 (N)
    transmittance = torch.exp(-transmittance)  # [..., "num_samples"]
    # transmittance = torch.cat([transmittance, torch.zeros([transmittance.shape[0], 1, 1], device=densities.device)], dim=1)
    weights = alphas * transmittance  # [..., "num_samples"]
    weights = torch.nan_to_num(weights)

    return weights, densities, transmittance

def get_density_linear(self, ray_samples: RaySamples) -> Tuple[Tensor, Tensor]:
    if self.use_integrated_encoding:
        gaussian_samples = ray_samples.frustums.get_gaussian_blob()
        if self.spatial_distortion is not None:
            gaussian_samples = self.spatial_distortion(gaussian_samples)
        encoded_xyz = self.position_encoding(gaussian_samples.mean, covs=gaussian_samples.cov)
    else:
        positions = torch.cat([ray_samples.frustums.origins + ray_samples.frustums.directions * ray_samples.frustums.starts, ray_samples.frustums.origins[..., -1:, :] + ray_samples.frustums.directions[..., -1:, :] * ray_samples.frustums.ends[..., -1:, :]], dim=-2)
        if self.spatial_distortion is not None:
            positions = self.spatial_distortion(positions)
        encoded_xyz = self.position_encoding(positions)
    base_mlp_out = self.mlp_base(encoded_xyz)
    density = self.field_output_density(base_mlp_out)
    return density, base_mlp_out

def get_outputs_linear(
        self, ray_samples: RaySamples, density_embedding: Optional[Tensor] = None
    ) -> Dict[FieldHeadNames, Tensor]:
        outputs = {}
        for field_head in self.field_heads:
            directions = F.pad(ray_samples.frustums.directions, (0, 0, 0, 1), mode='replicate')
            encoded_dir = self.direction_encoding(directions)
            mlp_out = self.mlp_head(torch.cat([encoded_dir, density_embedding], dim=-1))  # type: ignore
            outputs[field_head.field_head_name] = field_head(mlp_out)
        return outputs
    
# def pytorch_fwd_unscaled(
#         self,
#         in_tensor: Float[Tensor, "*bs input_dim"],
#         covs: Optional[Float[Tensor, "*bs input_dim input_dim"]] = None,
#     ) -> Float[Tensor, "*bs output_dim"]:
#         """Calculates NeRF encoding. If covariances are provided the encodings will be integrated as proposed
#             in mip-NeRF.

#         Args:
#             in_tensor: For best performance, the input tensor should be between 0 and 1.
#             covs: Covariances of input points.
#         Returns:
#             Output values will be between -1 and 1
#         """
#         # scaled_in_tensor = 2 * torch.pi * in_tensor  # scale to [0, 2pi]
#         scaled_in_tensor = in_tensor  # scale to [0, 2pi]
#         freqs = 2 ** torch.linspace(self.min_freq, self.max_freq, self.num_frequencies).to(in_tensor.device)
#         scaled_inputs = scaled_in_tensor[..., None, :] * freqs.reshape(-1, 1)  # [..., "num_scales", "input_dim"]
#         # scaled_inputs = scaled_inputs.view(*scaled_inputs.shape[:-2], -1)  # [..., "input_dim" * "num_scales"]

#         if covs is None:
#             # [..., "input_dim", "num_scales", 2]
#             encoded_inputs = torch.stack([torch.sin(scaled_inputs), torch.cos(scaled_inputs)], dim=-2)
#             encoded_inputs = encoded_inputs.reshape(*scaled_inputs.shape[:-2], -1)
#         else:
#             print(covs.shape)
#             input_var = torch.diagonal(covs, dim1=-2, dim2=-1)[..., :, None] * freqs[None, :] ** 2
#             # input_var = input_var.reshape((*input_var.shape[:-2], -1))
#             print(input_var.shape, scaled_inputs.shape)
#             encoded_inputs = expected_sin(
#                 torch.cat([scaled_inputs, scaled_inputs + torch.pi / 2.0], dim=-1), torch.cat(2 * [input_var], dim=-1)
#             )

#         if self.include_input:
#             encoded_inputs = torch.cat([in_tensor, encoded_inputs], dim=-1)
#         return encoded_inputs