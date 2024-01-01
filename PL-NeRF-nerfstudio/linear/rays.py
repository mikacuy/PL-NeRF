import random
from dataclasses import dataclass, field
from typing import Callable, Dict, Literal, Optional, Tuple, Union, overload, List

import torch
from jaxtyping import Float, Int, Shaped
from torch import Tensor

from nerfstudio.cameras.rays import RaySamples, Frustums
from nerfstudio.utils.tensor_dataclass import TensorDataclass

TORCH_DEVICE = Union[str, torch.device]



@dataclass
class RaySamplesLinear(RaySamples):
    """Samples along a ray"""

    """Times at which rays are sampled"""

    

    def get_weights(self, densities: Float[Tensor, "*batch num_samples 1"]) -> Float[Tensor, "*batch num_samples 1"]:
        """Return weights based on predicted densities

        Args:
            densities: Predicted densities for samples along ray

        Returns:
            Weights for each sample
        """

        delta_density = self.deltas * densities
        alphas = 1 - torch.exp(-delta_density)

        transmittance = torch.cumsum(delta_density[..., :-1, :], dim=-2)
        transmittance = torch.cat(
            [torch.zeros((*transmittance.shape[:1], 1, 1), device=densities.device), transmittance], dim=-2
        )
        transmittance = torch.exp(-transmittance)  # [..., "num_samples"]

        weights = alphas * transmittance  # [..., "num_samples"]
        weights = torch.nan_to_num(weights)

        return weights

    @overload
    @staticmethod
    def get_weights_and_transmittance_from_alphas(
        alphas: Float[Tensor, "*batch num_samples 1"], weights_only: Literal[True]
    ) -> Float[Tensor, "*batch num_samples 1"]:
        ...

    @overload
    @staticmethod
    def get_weights_and_transmittance_from_alphas(
        alphas: Float[Tensor, "*batch num_samples 1"], weights_only: Literal[False] = False
    ) -> Tuple[Float[Tensor, "*batch num_samples 1"], Float[Tensor, "*batch num_samples 1"]]:
        ...

    @staticmethod
    def get_weights_and_transmittance_from_alphas(
        alphas: Float[Tensor, "*batch num_samples 1"], weights_only: bool = False
    ) -> Union[
        Float[Tensor, "*batch num_samples 1"],
        Tuple[Float[Tensor, "*batch num_samples 1"], Float[Tensor, "*batch num_samples 1"]],
    ]:
        """Return weights based on predicted alphas
        Args:
            alphas: Predicted alphas (maybe from sdf) for samples along ray
            weights_only: If function should return only weights
        Returns:
            Tuple of weights and transmittance for each sample
        """

        transmittance = torch.cumprod(
            torch.cat([torch.ones((*alphas.shape[:1], 1, 1), device=alphas.device), 1.0 - alphas + 1e-7], 1), 1
        )

        weights = alphas * transmittance[:, :-1, :]
        if weights_only:
            return weights
        return weights, transmittance
    
    @classmethod
    def cat_samples(cls, ray_samples_list:List["RaySamples"]) -> "RaySamples":
        first_sample = ray_samples_list[0]
        combined_frustums = Frustums.cat_frustums([ray_samples.frustums for ray_samples in ray_samples_list])
        combined_metadata: Optional[Dict[str, Shaped[Tensor, "*bs latent_dims"]]] = None
        if first_sample.metadata is not None: 
            combined_metadata = {}
            for k in ray_samples_list[0].metadata.keys():
                combined_metadata[k] = torch.cat([vv.metadata[k] for vv in ray_samples_list])
        combined_sample = RaySamples(
            frustums=combined_frustums,
            metadata=combined_metadata,
            camera_indices = torch.cat([vv.camera_indices for vv in ray_samples_list]) if first_sample.camera_indices is not None else None,
            deltas = torch.cat([vv.deltas for vv in ray_samples_list]) if first_sample.deltas is not None else None,
            spacing_starts = torch.cat([vv.spacing_starts for vv in ray_samples_list]) if first_sample.spacing_starts is not None else None,
            spacing_ends = torch.cat([vv.spacing_ends for vv in ray_samples_list]) if first_sample.spacing_ends is not None else None,
            times = torch.cat([vv.times for vv in ray_samples_list]) if first_sample.times is not None else None,
            spacing_to_euclidean_fn=first_sample.spacing_to_euclidean_fn,
        )
        return combined_sample