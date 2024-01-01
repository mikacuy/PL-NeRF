from typing import Generator, Literal, Optional, Union

import torch
from jaxtyping import Float, Int
from torch import Tensor, nn

from nerfstudio.model_components.renderers import RGBRenderer, BackgroundColor



class LinearRGBRenderer(RGBRenderer):
    """Standard volumetric rendering.

    Args:
        background_color: Background color as RGB. Uses random colors if None.
    """
    def __init__(self, background_color: BackgroundColor = "random",  color_mode: Literal["left", "midpoint"]="midpoint", farcolorfix: bool = False, concat_walls: bool = True) -> None:
        super().__init__(background_color=background_color)
        self.color_mode=color_mode
        self.farcolorfix = farcolorfix

    def combine_rgb(
        self,
        rgb: Float[Tensor, "*bs num_samples 3"],
        weights: Float[Tensor, "*bs num_samples 1"],
        background_color: BackgroundColor = "random",
        ray_indices: Optional[Int[Tensor, "num_samples"]] = None,
        num_rays: Optional[int] = None,
    ) -> Float[Tensor, "*bs 3"]:
        """Composite samples along ray and render color image

        Args:
            rgb: RGB for each sample
            weights: Weights for each sample
            background_color: Background color as RGB.
            ray_indices: Ray index for each sample, used when samples are packed.
            num_rays: Number of rays, used when samples are packed.

        Returns:
            Outputs rgb values.
        """
        
        if ray_indices is not None and num_rays is not None:
            return super().combine_rgb(rgb, weights, background_color, ray_indices, num_rays)
        
        if self.color_mode == "midpoint":
            rgb = 0.5 * (rgb[:, 1:] + rgb[:, :-1])
            
        return RGBRenderer.combine_rgb(rgb, weights, background_color, ray_indices, num_rays)