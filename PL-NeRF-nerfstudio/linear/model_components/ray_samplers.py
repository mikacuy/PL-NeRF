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
Collection of sampling strategies
"""

from abc import abstractmethod
from typing import Any, Callable, List, Optional, Protocol, Tuple, Union, Dict, Literal

import torch
from jaxtyping import Float
from nerfacc import OccGridEstimator
from torch import Tensor, nn

from nerfstudio.cameras.rays import Frustums, RayBundle, RaySamples
from nerfstudio.model_components.ray_samplers import SpacedSampler, PDFSampler, ProposalNetworkSampler, Sampler

class LinearSpacedSampler(SpacedSampler):
    """Sample points according to a function.

    Args:
        num_samples: Number of samples per ray
        spacing_fn: Function that dictates sample spacing (ie `lambda x : x` is uniform).
        spacing_fn_inv: The inverse of spacing_fn.
        train_stratified: Use stratified sampling during training. Defaults to True
        single_jitter: Use a same random jitter for all samples along a ray. Defaults to False
    """

    def __init__(
        self,
        spacing_fn: Callable,
        spacing_fn_inv: Callable,
        num_samples: Optional[int] = None,
        train_stratified=True,
        single_jitter=False,
    ) -> None:
        super().__init__(num_samples=num_samples, spacing_fn=spacing_fn, spacing_fn_inv=spacing_fn_inv, train_stratified=train_stratified, single_jitter=single_jitter)

    def generate_ray_samples(
        self,
        ray_bundle: Optional[RayBundle] = None,
        num_samples: Optional[int] = None,
    ) -> RaySamples:
        """Generates position samples according to spacing function.

        Args:
            ray_bundle: Rays to generate samples for
            num_samples: Number of samples per ray

        Returns:
            Positions and deltas for samples along a ray
        """
        return super().generate_ray_samples(ray_bundle, num_samples, sample_method="piecewise_linear")


class LinearUniformSampler(LinearSpacedSampler):
    """Sample uniformly along a ray

    Args:
        num_samples: Number of samples per ray
        train_stratified: Use stratified sampling during training. Defaults to True
        single_jitter: Use a same random jitter for all samples along a ray. Defaults to False
    """

    def __init__(
        self,
        num_samples: Optional[int] = None,
        train_stratified=True,
        single_jitter=False,
    ) -> None:
        super().__init__(
            num_samples=num_samples,
            spacing_fn=lambda x: x,
            spacing_fn_inv=lambda x: x,
            train_stratified=train_stratified,
            single_jitter=single_jitter,
        )


class LinearLinearDisparitySampler(LinearSpacedSampler):
    """Sample linearly in disparity along a ray

    Args:
        num_samples: Number of samples per ray
        train_stratified: Use stratified sampling during training. Defaults to True
        single_jitter: Use a same random jitter for all samples along a ray. Defaults to False
    """

    def __init__(
        self,
        num_samples: Optional[int] = None,
        train_stratified=True,
        single_jitter=False,
    ) -> None:
        super().__init__(
            num_samples=num_samples,
            spacing_fn=lambda x: 1 / x,
            spacing_fn_inv=lambda x: 1 / x,
            train_stratified=train_stratified,
            single_jitter=single_jitter,
        )


class LinearSqrtSampler(LinearSpacedSampler):
    """Square root sampler along a ray

    Args:
        num_samples: Number of samples per ray
        train_stratified: Use stratified sampling during training. Defaults to True
    """

    def __init__(
        self,
        num_samples: Optional[int] = None,
        train_stratified=True,
        single_jitter=False,
    ) -> None:
        super().__init__(
            num_samples=num_samples,
            spacing_fn=torch.sqrt,
            spacing_fn_inv=lambda x: x**2,
            train_stratified=train_stratified,
            single_jitter=single_jitter,
        )


class LinearLogSampler(LinearSpacedSampler):
    """Log sampler along a ray

    Args:
        num_samples: Number of samples per ray
        train_stratified: Use stratified sampling during training. Defaults to True
    """

    def __init__(
        self,
        num_samples: Optional[int] = None,
        train_stratified=True,
        single_jitter=False,
    ) -> None:
        super().__init__(
            num_samples=num_samples,
            spacing_fn=torch.log,
            spacing_fn_inv=torch.exp,
            train_stratified=train_stratified,
            single_jitter=single_jitter,
        )





class LinearUniformLinDispPiecewiseSampler(LinearSpacedSampler):
    """Piecewise sampler along a ray that allocates the first half of the samples uniformly and the second half
    using linearly in disparity spacing.


    Args:
        num_samples: Number of samples per ray
        train_stratified: Use stratified sampling during training. Defaults to True
        single_jitter: Use a same random jitter for all samples along a ray. Defaults to False
    """

    def __init__(
        self,
        num_samples: Optional[int] = None,
        train_stratified=True,
        single_jitter=False,
    ) -> None:
        super().__init__(
            num_samples=num_samples,
            spacing_fn=lambda x: torch.where(x < 1, x / 2, 1 - 1 / (2 * x)),
            spacing_fn_inv=lambda x: torch.where(x < 0.5, 2 * x, 1 / (2 - 2 * x)),
            train_stratified=train_stratified,
            single_jitter=single_jitter,
        )



def pw_linear_sample_increasing(s_left, s_right, T_left, tau_left, tau_right, u, epsilon=1e-3):
    ### Fix this, need negative sign
    ln_term = -torch.log(torch.max(torch.ones_like(T_left)*epsilon, torch.div(1-u, torch.max(torch.ones_like(T_left)*epsilon,T_left) ) ))
    discriminant = tau_left**2 + torch.div( 2 * (tau_right - tau_left) * ln_term , torch.max(torch.ones_like(s_right)*epsilon, s_right - s_left) )

    t = torch.div( (s_right - s_left) * (-tau_left + torch.sqrt(torch.max(torch.ones_like(discriminant)*epsilon, discriminant))) , torch.max(torch.ones_like(tau_left)*epsilon, tau_right - tau_left))

    ### clamp t to [0, s_right - s_left]
    # print("t clamping")
    # print(torch.max(t))
    t = torch.clamp(t, torch.ones_like(t, device=t.device)*epsilon, s_right - s_left)
    # print(torch.max(t))
    # print()

    sample = s_left + t

    return sample

def pw_linear_sample_decreasing(s_left, s_right, T_left, tau_left, tau_right, u, epsilon=1e-3):
    ### Fix this, need negative sign
    ln_term = -torch.log(torch.max(torch.ones_like(T_left)*epsilon, torch.div(1-u, torch.max(torch.ones_like(T_left)*epsilon,T_left) ) ))
    discriminant = tau_left**2 - torch.div( 2 * (tau_left - tau_right) * ln_term , torch.max(torch.ones_like(s_right)*epsilon, s_right - s_left) )
    t = torch.div( (s_right - s_left) * (tau_left - torch.sqrt(torch.max(torch.ones_like(discriminant)*epsilon, discriminant))) , torch.max(torch.ones_like(tau_left)*epsilon, tau_left - tau_right))

    ### clamp t to [0, s_right - s_left]
    # print("t clamping")
    # print(torch.max(t))
    t = torch.clamp(t, torch.ones_like(t, device=t.device)*epsilon, s_right - s_left)
    # print(torch.max(t))
    # print()

    sample = s_left + t

    return sample
def pw_linear_sample_fn(cdf, existing_bins, u, densities, transmittance, epsilon=1e-3, zero_threshold=1e-4):
    transmittance = torch.cat([transmittance, torch.zeros([transmittance.shape[0], 1, 1], device=densities.device)], dim=1)
    inds = torch.searchsorted(cdf, u, side="right")
    below = torch.clamp(inds - 1, 0, existing_bins.shape[-1] - 1)
    above = torch.clamp(inds, 0, existing_bins.shape[-1] - 1)
    bins_g0 = torch.gather(existing_bins, -1, below)
    bins_g1 = torch.gather(existing_bins, -1, above)
    densities_g0 = torch.gather(densities[..., 0], -1, below)
    densities_g1 = torch.gather(densities[..., 0], -1, above)
    transmittance_g0 = torch.gather(transmittance[..., 0], -1, below)
    densities_diff = densities[:, 1:] - densities[:, :-1]
    densities_diff_g0 = torch.gather(densities_diff[..., 0], -1, torch.clamp(below, 0, densities_diff.shape[-1] - 1))
    
    dummy = torch.ones_like(bins_g0) * -1.0

    ### Constant interval, take the left bin
    samples1 = torch.where(torch.logical_and(densities_diff_g0 < zero_threshold, densities_diff_g0 > -zero_threshold), bins_g0, dummy)

    samples2 = torch.where(bins_g0 >= zero_threshold, pw_linear_sample_increasing(bins_g0, bins_g1, transmittance_g0, densities_g0, densities_g1, u, epsilon=epsilon), samples1)

    samples3 = torch.where(bins_g0 <= -zero_threshold, pw_linear_sample_decreasing(bins_g0, bins_g1, transmittance_g0, densities_g0, densities_g1, u, epsilon=epsilon), samples2)


    ## Check for nan --> need to figure out why
    samples = torch.where(torch.isnan(samples3), bins_g0, samples3)


    return samples

class LinearPDFSampler(PDFSampler):
    """Sample based on probability distribution

    Args:
        num_samples: Number of samples per ray
        train_stratified: Randomize location within each bin during training.
        single_jitter: Use a same random jitter for all samples along a ray. Defaults to False
        include_original: Add original samples to ray.
        histogram_padding: Amount to weights prior to computing PDF.
    """

    def __init__(
        self,
        num_samples: Optional[int] = None,
        train_stratified: bool = True,
        single_jitter: bool = False,
        include_original: bool = True,
        histogram_padding: float = 0.01,
        concat_walls: bool = True
    ) -> None:
        super().__init__(num_samples=num_samples, train_stratified=train_stratified, single_jitter=single_jitter, include_original=include_original, histogram_padding=histogram_padding)
        self.concat_walls=concat_walls



    def generate_ray_samples(
        self,
        ray_bundle: Optional[RayBundle] = None,
        ray_samples: Optional[RaySamples] = None,
        weights: Optional[Float[Tensor, "*batch num_samples 1"]] = None,
        densities: Optional[Float[Tensor, "*batch num_samples 1"]] = None,
        transmittance: Optional[Float[Tensor, "*batch num_samples 1"]] = None,
        num_samples: Optional[int] = None,
        eps: float = 1e-5,
    ) -> RaySamples:
        """Generates position samples given a distribution.

        Args:
            ray_bundle: Rays to generate samples for
            ray_samples: Existing ray samples
            weights: Weights for each bin
            num_samples: Number of samples per ray
            eps: Small value to prevent numerical issues.

        Returns:
            Positions and deltas for samples along a ray
        """

        if ray_samples is None or ray_bundle is None:
            raise ValueError("ray_samples and ray_bundle must be provided")
        assert weights is not None, "weights must be provided"

        num_samples = num_samples or self.num_samples
        assert num_samples is not None
        num_bins = num_samples + 1

        weights = weights[..., 0] + self.histogram_padding

        # Add small offset to rays with zero weight to prevent NaNs
        weights_sum = torch.sum(weights, dim=-1, keepdim=True)
        padding = torch.relu(eps - weights_sum)
        weights = weights + padding / weights.shape[-1]
        weights_sum += padding
        pdf = weights / weights.sum(dim=-1, keepdim=True)
        cdf = torch.min(torch.ones_like(pdf), torch.cumsum(pdf, dim=-1))
        cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], dim=-1) # N + 3 (N + 1)
        if self.train_stratified and self.training:
            # Stratified samples between 0 and 1
            u = torch.linspace(0.0, 1.0 - (1.0 / num_bins), steps=num_bins, device=cdf.device)
            u = u.expand(size=(*cdf.shape[:-1], num_bins))
            if self.single_jitter:
                rand = torch.rand((*cdf.shape[:-1], 1), device=cdf.device) / num_bins
            else:
                rand = torch.rand((*cdf.shape[:-1], num_samples + 1), device=cdf.device) / num_bins
            u = u + rand
        else:
            # Uniform samples between 0 and 1
            u = torch.linspace(0.0, 1.0 - (1.0 / num_bins), steps=num_bins, device=cdf.device)
            u = u + 1.0 / (2 * num_bins)
            u = u.expand(size=(*cdf.shape[:-1], num_bins))
        u = u.contiguous()

        assert (
            ray_samples.spacing_starts is not None and ray_samples.spacing_ends is not None
        ), "ray_sample spacing_starts and spacing_ends must be provided"
        assert ray_samples.spacing_to_euclidean_fn is not None, "ray_samples.spacing_to_euclidean_fn must be provided"
        if self.concat_walls:
            existing_bins = torch.cat(
                [
                    torch.zeros_like(ray_samples.spacing_starts[..., :1, 0]),
                    ray_samples.spacing_starts[..., 0],
                    ray_samples.spacing_ends[..., -1:, 0], 
                    torch.ones_like(ray_samples.spacing_starts[..., :1, 0]), # N + 3
                ],
                dim=-1,
            )
        else:
            existing_bins = torch.cat(
                [
                    ray_samples.spacing_starts[..., 0],
                    ray_samples.spacing_ends[..., -1:, 0], 
                ],
                dim=-1,
            )
        

        bins = pw_linear_sample_fn(cdf, existing_bins, u, densities, transmittance) 

        if self.include_original:
            bins, _ = torch.sort(torch.cat([existing_bins[..., 1:], bins], -1), -1)

        # Stop gradients
        bins = bins.detach()

        euclidean_bins = ray_samples.spacing_to_euclidean_fn(bins)

        ray_samples = ray_bundle.get_ray_samples(
            bin_starts=euclidean_bins[..., :-1, None],
            bin_ends=euclidean_bins[..., 1:, None],
            spacing_starts=bins[..., :-1, None],
            spacing_ends=bins[..., 1:, None],
            spacing_to_euclidean_fn=ray_samples.spacing_to_euclidean_fn)
        return ray_samples
    
    
class LinearProposalNetworkSampler(ProposalNetworkSampler):
    def __init__(
        self,
        num_proposal_samples_per_ray: Tuple[int, ...] = (64,),
        num_nerf_samples_per_ray: int = 32,
        num_proposal_network_iterations: int = 2,
        single_jitter: bool = False,
        update_sched: Callable = lambda x: 1,
        initial_sampler: Optional[Sampler] = None,
        add_end_bin: bool = False
    ) -> None:
        super().__init__(
            num_proposal_samples_per_ray=num_proposal_samples_per_ray,
            num_nerf_samples_per_ray=num_nerf_samples_per_ray,
            num_proposal_network_iterations=num_proposal_network_iterations,
            single_jitter=single_jitter,
            update_sched=update_sched,
            initial_sampler=initial_sampler,
            add_end_bin=False)

        # samplers
        if initial_sampler is None:
            self.initial_sampler = LinearUniformLinDispPiecewiseSampler(single_jitter=single_jitter)
        else:
            self.initial_sampler = initial_sampler
        self.pdf_sampler = LinearPDFSampler(include_original=False, single_jitter=single_jitter, add_end_bin=add_end_bin)

    def generate_ray_samples(
        self,
        ray_bundle: Optional[RayBundle] = None,
        density_fns: Optional[List[Callable]] = None,
        **density_kwargs
    ) -> Tuple[RaySamples, List, List]:
        assert ray_bundle is not None
        assert density_fns is not None

        weights_list = []
        ray_samples_list = []

        n = self.num_proposal_network_iterations
        weights = None
        ray_samples: Optional[RaySamples] = None
        updated = self._steps_since_update > self.update_sched(self._step) or self._step < 10
        for i_level in range(n + 1):
            is_prop = i_level < n
            num_samples = self.num_proposal_samples_per_ray[i_level] if is_prop else self.num_nerf_samples_per_ray
            if i_level == 0:
                # Uniform sampling because we need to start with some samples
                ray_samples = self.initial_sampler(ray_bundle, num_samples=num_samples)
            else:
                # PDF sampling based on the last samples and their weights
                # Perform annealing to the weights. This will be a no-op if self._anneal is 1.0.
                assert weights is not None
                ray_samples = self.pdf_sampler(ray_bundle, ray_samples, weights, densities, transmittance, num_samples=num_samples)
            if is_prop:
                assert ray_samples is not None
                if updated:
                    # always update on the first step or the inf check in grad scaling crashes
                    density = density_fns[i_level](ray_samples.frustums.get_positions())
                else:
                    with torch.no_grad():
                        density = density_fns[i_level](ray_samples.frustums.get_positions())
                weights, densities, transmittance = ray_samples.get_weights_linear(density)
                weights_list.append(weights[:, 1:-1])  # (num_rays, num_samples)
                ray_samples_list.append(ray_samples[:, :-1])
        if updated:
            self._steps_since_update = 0

        assert ray_samples is not None
        return ray_samples, weights_list, ray_samples_list
