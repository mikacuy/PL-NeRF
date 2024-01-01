
from nerfstudio.data.dataparsers.blender_dataparser import BlenderDataParserConfig
from nerfstudio.data.datamanagers.base_datamanager import VanillaDataManagerConfig
from nerfstudio.plugins.types import MethodSpecification
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.pipelines.base_pipeline import VanillaPipelineConfig
from nerfstudio.data.dataparsers.nerfstudio_dataparser import NerfstudioDataParserConfig
from linear.models.linear_mipnerf import LinearMipNerfModelConfig
from linear.schedulers import LogDecaySchedulerConfig
from linear.optimizers import AdamWOptimizerConfig
linear_mip_nerf = MethodSpecification(
    TrainerConfig(
    method_name="linear-mipnerf",
    steps_per_eval_image=5000,
    max_num_iterations=1000001,
    pipeline=VanillaPipelineConfig(
        datamanager=VanillaDataManagerConfig(dataparser=BlenderDataParserConfig(), train_num_rays_per_batch=4096),
        model=LinearMipNerfModelConfig(
            loss_coefficients={"rgb_loss_coarse": 0.1, "rgb_loss_fine": 1.0},
            num_coarse_samples=128,
            num_importance_samples=128,
            eval_num_rays_per_chunk=1024,
            collider_params=dict(near_plane=2.0, far_plane=6.0),
            color_mode="midpoint"
        ),
    ),
    optimizers={
        "fields": {
            "optimizer": AdamWOptimizerConfig(lr=5e-4, weight_decay=1e-5),
            "scheduler": LogDecaySchedulerConfig(lr_final=1e-6, lr_delay_steps=2500, lr_delay_mult=0.01, max_steps=1000000),
        }
    }),
    description="linear mip model"
)